#!/usr/bin/env python3
"""Evaluate segmented VO independently of estimation; no reference feedback.
KITTI: ATE/RPE relative to GT. Arena: disagreement with FAST-LIO, not accuracy.
"""
import argparse
from collections import Counter, deque
import csv
import json
from pathlib import Path
import cv2 as cv
import numpy as np


def quaternion_matrix(q):
    q=np.asarray(q,float);n=np.linalg.norm(q)
    if not np.isfinite(n) or n<1e-10:raise ValueError('invalid quaternion')
    x,y,z,w=q/n
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def tf_chain(path,source,target):
    """Return T_source_target: a point in target expressed in source."""
    graph={}
    for r in json.loads(Path(path).read_text()):
        p,c=r['parent'].lstrip('/'),r['child'].lstrip('/')
        T=np.eye(4);T[:3,:3]=quaternion_matrix(r['quaternion_xyzw']);T[:3,3]=r['translation']
        graph.setdefault(p,[]).append((c,T));graph.setdefault(c,[]).append((p,np.linalg.inv(T)))
    queue=deque([(source.lstrip('/'),np.eye(4),[])])
    seen=set()
    while queue:
        name,T,route=queue.popleft()
        if name==target.lstrip('/'):return T,route+[name]
        if name in seen:continue
        seen.add(name)
        for other,edge in graph.get(name,[]):queue.append((other,T@edge,route+[name]))
    raise ValueError(f'no recorded TF path from {source} to {target}')


def load_reference(path,transform=None):
    rows=list(csv.DictReader(open(path)));ts=[];poses=[]
    for r in rows:
        if 'r00' in r:R=np.array([float(r[f'r{i}{j}']) for i in range(3) for j in range(3)]).reshape(3,3)
        else:R=quaternion_matrix([float(r[k]) for k in ['qx','qy','qz','qw']])
        T=np.eye(4);T[:3,:3]=R;T[:3,3]=[float(r[k]) for k in ['x','y','z']]
        if transform is not None:T=T@transform
        ts.append(float(r['timestamp']));poses.append(T)
    ts=np.array(ts);poses=np.array(poses)
    if len(ts)<2 or not np.isfinite(ts).all() or not np.all(np.diff(ts)>0):raise ValueError('reference timestamps must increase; at least 2 poses')
    return ts,poses


def interpolate(ts,poses,t,max_gap=.35):
    j=int(np.searchsorted(ts,t))
    if j<len(ts) and abs(ts[j]-t)<1e-8:return poses[j]
    if j==0 or j==len(ts) or ts[j]-ts[j-1]>max_gap:return None
    f=(t-ts[j-1])/(ts[j]-ts[j-1]);A,B=poses[j-1],poses[j]
    # Geodesic SO(3) interpolation (equivalent to quaternion SLERP).
    rv=cv.Rodrigues(A[:3,:3].T@B[:3,:3])[0]
    T=np.eye(4);T[:3,:3]=A[:3,:3]@cv.Rodrigues(f*rv)[0];T[:3,3]=(1-f)*A[:3,3]+f*B[:3,3]
    return T


def align(x,y,with_scale):
    xc,yc=x-x.mean(0),y-y.mean(0);variance=np.sum(xc*xc)/len(x)
    if variance<1e-8:raise ValueError('insufficient motion for trajectory alignment')
    U,S,Vt=np.linalg.svd(yc.T@xc/len(x));D=np.eye(3);D[2,2]=np.linalg.det(U@Vt)
    R=U@D@Vt;scale=float(np.sum(S*np.diag(D))/variance) if with_scale else 1.
    if scale<=0:raise ValueError('invalid alignment scale')
    t=y.mean(0)-scale*R@x.mean(0)
    return scale,R,t


def angle(R):return float(np.linalg.norm(cv.Rodrigues(R)[0])*180/np.pi)


def run(trajectory,reference,mode,reference_kind='ground_truth',tf_static=None,source_frame='kalibr/livox/imu',camera_frame='kalibr/rgbd/infra1_optical',rpe_dt=1.):
    rows=list(csv.DictReader(open(trajectory)));transform=None;route=None
    if tf_static:transform,route=tf_chain(tf_static,source_frame,camera_frame)
    ts,gt=load_reference(reference,transform)
    summary=dict(mode=mode,reference_kind=reference_kind,
                 meaning='error against ground truth' if reference_kind=='ground_truth' else 'disagreement with FAST-LIO, NOT accuracy',
                 alignment='Sim(3), one global scale per independent segment' if mode=='mono' else 'SE(3), rigid, scale FIXED to 1',
                 reference_used_for_estimation=False,reference_tf_route=route,
                 reference_tf_note='Recorded nominal static transforms; physical camera-LiDAR calibration not independently verified' if route else None,
                 frame_count=len(rows),accepted_count=sum(int(r['accepted']) for r in rows),
                 states=dict(Counter(r['state'] for r in rows)),segments=[])
    aligned=[]
    for segment in sorted(set(int(r['segment']) for r in rows)):
        selected=[r for r in rows if int(r['segment'])==segment and int(r['accepted']) and int(r['pose_valid'])]
        pairs=[]
        for r in selected:
            ref=interpolate(ts,gt,float(r['timestamp']))
            if ref is not None:
                T=np.eye(4);T[:3,3]=[float(r[k]) for k in ['x','y','z']];T[:3,:3]=np.array([float(r[f'r{i}{j}']) for i in range(3) for j in range(3)]).reshape(3,3)
                pairs.append((float(r['timestamp']),T,ref))
        metric=dict(segment=segment,accepted=len(selected),paired=len(pairs))
        if len(pairs)<8:
            metric['status']='too few paired poses';summary['segments'].append(metric);continue
        times=np.array([p[0] for p in pairs]);est=np.array([p[1] for p in pairs]);ref=np.array([p[2] for p in pairs]);x,y=est[:,:3,3],ref[:,:3,3]
        try:s,R,t=align(x,y,mode=='mono')
        except ValueError as err:
            metric['status']=str(err);summary['segments'].append(metric);continue
        fit=s*(x@R.T)+t;errors=np.linalg.norm(fit-y,axis=1)
        # Scale translations only. Rotations retain SO(3), also for Sim(3).
        fitposes=est.copy();fitposes[:,:3,3]=fit;fitposes[:,:3,:3]=R@est[:,:3,:3]
        trans,rot=[],[]
        for i in range(len(times)):
            j=int(np.searchsorted(times,times[i]+rpe_dt))
            if j>=len(times) or abs(times[j]-times[i]-rpe_dt)>.08:continue
            # Do not evaluate relative motion across rejected tracking gaps.
            if np.max(np.diff(times[i:j+1]))>.35:continue
            d_est=np.linalg.inv(fitposes[i])@fitposes[j];d_ref=np.linalg.inv(ref[i])@ref[j]
            delta=np.linalg.inv(d_ref)@d_est;trans.append(np.linalg.norm(delta[:3,3]));rot.append(angle(delta[:3,:3]))
        metric.update(status='evaluated',alignment_scale=s,position_rmse=float(np.sqrt(np.mean(errors**2))),position_p95=float(np.percentile(errors,95)),
                      rpe_dt_seconds=rpe_dt,rpe_pairs=len(trans),rpe_translation_rmse=float(np.sqrt(np.mean(np.square(trans)))) if trans else None,
                      rpe_rotation_rmse_deg=float(np.sqrt(np.mean(np.square(rot)))) if rot else None,
                      start_timestamp=float(times[0]),end_timestamp=float(times[-1]))
        for j in range(len(times)):aligned.append([times[j],segment,*fit[j],*y[j],errors[j]])
        summary['segments'].append(metric)
    return summary,aligned


def plot_diagnostics(trajectory,aligned,summary,path):
    import os
    cache=Path(path).resolve().parent/".matplotlib-cache"
    cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR",str(cache))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=list(csv.DictReader(open(trajectory)));fig,axes=plt.subplots(2,2,figsize=(12,8))
    a=np.array(aligned)
    if len(a):
        for seg in sorted(set(a[:,1])):
            b=a[a[:,1]==seg];axes[0,0].plot(b[:,2],b[:,4],label=f'VO seg {int(seg)}')
            axes[0,0].plot(b[:,5],b[:,7],'--',alpha=.6,label=f'ref seg {int(seg)}')
            axes[0,1].plot(b[:,0]-a[0,0],b[:,8],label=f'seg {int(seg)}')
        axes[0,0].legend(fontsize=7);axes[0,0].axis('equal')
    axes[0,0].set(xlabel='x [reference unit]',ylabel='z [reference unit]',title='Trajectory after evaluation alignment')
    axes[0,1].set(xlabel='time [s]',ylabel='position difference',title=summary['meaning'])
    t=np.array([float(r['timestamp']) for r in rows]);t-=t[0]
    axes[1,0].plot(t,[float(r['inliers']) for r in rows],label='inliers')
    axes[1,0].plot(t,[float(r['map_points']) for r in rows],label='map points');axes[1,0].legend();axes[1,0].set_xlabel('time [s]')
    axes[1,1].plot(t,[float(r['reprojection_px']) for r in rows],label='reprojection [px]')
    axes[1,1].plot(t,[float(r['epipolar_px']) for r in rows],label='epipolar [px]')
    axes[1,1].scatter(t,[int(r['accepted']) for r in rows],s=3,label='accepted');axes[1,1].legend();axes[1,1].set_xlabel('time [s]')
    fig.suptitle(f"{summary['mode']} | {summary['alignment']}",fontsize=10);fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trajectory',required=True);ap.add_argument('--reference',required=True);ap.add_argument('--mode',choices=['mono','stereo'],required=True)
    ap.add_argument('--reference-kind',choices=['ground_truth','fastlio'],default='ground_truth');ap.add_argument('--tf-static')
    ap.add_argument('--source-frame',default='kalibr/livox/imu');ap.add_argument('--camera-frame',default='kalibr/rgbd/infra1_optical');ap.add_argument('--output',required=True)
    a=ap.parse_args();s,points=run(a.trajectory,a.reference,a.mode,a.reference_kind,a.tf_static,a.source_frame,a.camera_frame)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.with_suffix('.json').write_text(json.dumps(s,indent=2,allow_nan=False)+'\n')
    with out.with_suffix('.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['timestamp','segment','x_vo','y_vo','z_vo','x_ref','y_ref','z_ref','position_difference']);w.writerows(points)
    plot_diagnostics(a.trajectory,points,s,out.with_suffix('.png'))
    print(json.dumps(s,indent=2))

if __name__=='__main__':main()
