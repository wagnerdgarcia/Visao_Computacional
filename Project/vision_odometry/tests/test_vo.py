import csv
import json
import sys
from pathlib import Path
import cv2 as cv
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
import vo

@pytest.fixture(autouse=True)
def deterministic():
    cv.setRNGSeed(42);cv.setNumThreads(1)

@pytest.fixture
def obj():return json.loads((ROOT/'config/synthetic.json').read_text())

@pytest.fixture
def scene():
    X=np.random.default_rng(17).uniform([-3,-2,4],[3,2,10],(300,3))
    K=np.array([[400.,0,320],[0,400,240],[0,0,1]])
    return X,K,cv.Rodrigues(np.array([.08,-.12,.03]))[0],np.array([-.45,.02,.08])

def project(X,K,R=np.eye(3),t=np.zeros(3)):
    return cv.projectPoints(X,cv.Rodrigues(R)[0],t,K,None)[0].reshape(-1,2)

def test_composition_rotated_camera(scene):
    _,_,R,t=scene;Rwc=cv.Rodrigues(np.array([.2,.4,-.3]))[0];C=np.array([2.,3.,4.]);X=np.array([1.,2.,7.])
    Rnew,Cnew=vo.compose(Rwc,C,R,t,2.3)
    assert np.allclose(Rnew@(R@(2.3*X)+2.3*t)+Cnew,Rwc@(2.3*X)+C)
    assert not np.allclose(Cnew,C+R@t)

def test_pnp_world_to_camera(scene):
    X,K,R,t=scene;z=vo.pnp(X,project(X,K,R,t),K,(640,480))
    assert z.accepted and z.inliers==len(X) and np.linalg.norm(z.R-R)<1e-6
    Rwc,C=vo.camera_pose(z.R,z.t)
    assert np.allclose(C,-R.T@t,atol=1e-6)
    assert np.allclose(Rwc@(R@X[0]+t)+C,X[0],atol=1e-6)

def test_bootstrap_gauge_and_map_pnp(scene):
    X,K,R,t=scene;z=vo.bootstrap(project(X,K),project(X,K,R,t),K,(640,480))
    assert z.accepted and np.linalg.norm(z.R-R)<1e-5 and np.dot(z.t,t/np.linalg.norm(t))>.999
    assert np.median(np.linalg.norm(z.X[z.mask]*np.linalg.norm(t)-X[z.mask],axis=1))<1e-4
    assert z.cheirality>.99
    R2=cv.Rodrigues(np.array([.1,-.15,.04]))[0];t2=np.array([-.72,.03,.1])
    z2=vo.pnp(z.X[z.mask],project(X[z.mask],K,R2,t2),K,(640,480))
    assert z2.accepted and np.allclose(z2.t,t2/np.linalg.norm(t),atol=1e-4)
    assert not np.isclose(np.linalg.norm(z2.t),1.)

@pytest.mark.parametrize('which',['pnp','essential'])
def test_outliers_retained_mask(scene,which):
    X,K,R,t=scene;a=project(X,K);b=project(X,K,R,t)
    b[:60]=np.random.default_rng(8).uniform([0,0],[640,480],(60,2))
    z=vo.pnp(X,b,K,(640,480)) if which=='pnp' else vo.bootstrap(a,b,K,(640,480))
    assert z.accepted and z.inliers>=220 and np.sum(z.mask[:60])<=2 and z.error<.1

def test_outlier_majority_rejected(scene):
    X,K,_,_=scene;b=np.random.default_rng(3).uniform([0,0],[640,480],(len(X),2))
    assert not vo.pnp(X,b,K,(640,480)).accepted
    assert not vo.bootstrap(project(X,K),b,K,(640,480)).accepted

def test_stereo_metric_depth_scale(scene):
    X,K,R,t=scene;baseline=np.array([-.2,0.,0.]);a,b=project(X,K),project(X,K,t=baseline)
    points,valid,e=vo.triangulate(a,b,K,np.eye(3),baseline,.5)
    assert valid.all() and np.max(e)<1e-8 and np.allclose(points,X,atol=1e-7)
    z=vo.pnp(points,project(X,K,R,t),K,(640,480))
    assert z.accepted and np.allclose(z.t,t,atol=1e-6)
    doubled,_,_=vo.triangulate(a,b,K,np.eye(3),2*baseline,.5)
    assert np.allclose(doubled,2*X,atol=1e-6)
    _,bad,_=vo.triangulate(a,b,K,np.eye(3),-baseline,.5)
    assert not bad.any()

def test_invalid_disparity_reprojection(scene):
    X,K,_,_=scene;a=project(X,K);b=project(X,K,t=np.array([-.2,0.,0.]));b[:40,1]+=20
    _,good,_=vo.triangulate(a,b,K,np.eye(3),[-.2,0,0],.5)
    assert not good[:40].any() and good[40:].all()
    _,good,_=vo.triangulate(a,a,K,np.eye(3),[-.2,0,0],.5)
    assert not good.any()

def test_rectification_nonidentity(scene,obj):
    X,K,_,_=scene;R=cv.Rodrigues(np.array([.01,-.02,.005]))[0];t=np.array([-.2,.001,.002])
    obj.update(already_rectified=False,R_right_left=R.tolist(),T_right_left=t.tolist());cal=vo.Calibration(obj,'stereo')
    R1,R2,P1,P2,*_=cv.stereoRectify(K,np.zeros(5),K,np.zeros(5),(640,480),R,t,flags=cv.CALIB_ZERO_DISPARITY,alpha=0)
    a=cv.undistortPoints(project(X,K).reshape(-1,1,2),K,None,R=R1,P=P1).reshape(-1,2)
    b=cv.undistortPoints(project(X,K,R,t).reshape(-1,1,2),K,None,R=R2,P=P2).reshape(-1,2)
    assert np.max(abs(a[:,1]-b[:,1]))<1e-7
    im=np.zeros((480,640),np.uint8)
    assert cal.prepare(im).shape==im.shape and cal.prepare(im,1).shape==im.shape

@pytest.mark.parametrize('kind',['stationary','rotation','planar','clustered'])
def test_degeneracy(scene,kind):
    X,K,R,t=scene
    if kind=='stationary':R,t=np.eye(3),np.zeros(3)
    if kind=='rotation':t=np.zeros(3)
    if kind=='planar':X=X.copy();X[:,2]=6
    if kind=='clustered':X=X.copy();X[:,:2]*=.01
    z=vo.bootstrap(project(X,K),project(X,K,R,t),K,(640,480))
    assert not z.accepted
    if kind=='stationary':assert z.state=='STATIONARY'
    if kind=='rotation':assert z.state=='ROTATION_ONLY'

def test_empty_descriptors_no_texture_few_points(obj):
    f=vo.detect(np.zeros((480,640),np.uint8));assert len(f.points)==0 and f.descriptors is None
    g=vo.Features(np.zeros((1,2)),np.zeros((1,128),np.float32))
    assert len(vo.match(f,g))==0 and len(vo.match(g,g))==0
    K=np.array(obj['K_left'],float)
    assert not vo.pnp(np.empty((0,3)),np.empty((0,2)),K,(640,480)).accepted
    assert not vo.bootstrap(np.zeros((5,2)),np.zeros((5,2)),K,(640,480)).accepted

@pytest.mark.parametrize('mutation',['resolution','unit','baseline','rotation','intrinsics','distortion','missing'])
def test_bad_calibration(obj,mutation):
    if mutation=='resolution':obj['width']=0
    if mutation=='unit':obj.pop('translation_unit')
    if mutation=='baseline':obj['T_right_left']=[0,0,0]
    if mutation=='rotation':obj['R_right_left'][0][0]=2
    if mutation=='intrinsics':obj['K_left'][0][0]=-1
    if mutation=='distortion':obj['D_right']=[0,0]
    if mutation=='missing':obj.pop('K_right')
    with pytest.raises((ValueError,KeyError)):vo.Calibration(obj,'stereo')

def test_resolution_missing_pair_no_update(obj):
    e=vo.Odometry(vo.Calibration(obj,'stereo'),'stereo')
    with pytest.raises(ValueError):e.step(np.zeros((20,20),np.uint8),np.zeros((20,20),np.uint8))
    with pytest.raises(ValueError):e.step(np.zeros((480,640),np.uint8))
    assert not e.has_pose and np.array_equal(e.C,np.zeros(3))

def test_lost_reset_no_rejected_update(obj):
    e=vo.Odometry(vo.Calibration(obj,'mono'),'mono');e.C=np.array([1.,2.,3.]);e.has_pose=True
    blank=np.zeros((480,640),np.uint8);before=e.C.copy()
    for _ in range(2):
        z=e.step(blank);assert not z.accepted and np.array_equal(e.C,before) and e.segment==0
    z=e.step(blank)
    assert z.state=='REINITIALIZING' and e.segment==1 and not e.has_pose and e.landmarks=={}
    texture=np.random.default_rng(4).integers(0,256,blank.shape,dtype=np.uint8)
    assert e.step(texture).state=='ANCHOR' and e.segment==1 and not e.has_pose
    assert e.step(texture).state=='STATIONARY' and not e.has_pose

def test_pipeline_associations_keyframes(scene,obj,monkeypatch):
    X,K,_,_=scene;d=np.random.default_rng(5).normal(size=(len(X),128)).astype(np.float32);frames=[];poses=[]
    for i in range(20):
        R=cv.Rodrigues(np.array([.002*i,-.006*i,0.]))[0];t=np.array([-.12*i,0.,.02*i])
        frames.append(vo.Features(project(X,K,R,t),d.copy()));poses.append((R,t))
    features=iter(frames);monkeypatch.setattr(vo,'detect',lambda _:next(features))
    e=vo.Odometry(vo.Calibration(obj,'mono'),'mono');blank=np.zeros((480,640),np.uint8)
    results=[e.step(blank) for _ in frames]
    assert results[0].state=='ANCHOR' and sum(z.accepted for z in results)>=18
    init=next(i for i,z in enumerate(results) if z.state=='INITIALIZED');gauge=np.linalg.norm(poses[init][1]);R,t=poses[-1]
    assert np.allclose(e.C,-R.T@t/gauge,atol=1e-4) and np.allclose(e.R,R.T,atol=1e-5)
    assert e.age<8 and len(e.landmarks)>=vo.MIN_POINTS and e.segment==0

def manifest(tmp_path,**updates):
    for x in ['l.png','r.png']:(tmp_path/x).write_bytes(b'placeholder')
    row=dict(index=0,timestamp='1.0',left='l.png',right='r.png',right_timestamp='1.0');row.update(updates);p=tmp_path/'frames.csv'
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=row.keys());w.writeheader();w.writerow(row)
    return p

@pytest.mark.parametrize('bad',[{'right':''},{'right':'absent.png'},{'right':'l.png'},{'right_timestamp':'1.1'},{'right_timestamp':'nan'}])
def test_manifest_invalid_pairs(tmp_path,bad):
    with pytest.raises(ValueError):list(vo.manifest_rows(manifest(tmp_path,**bad),'stereo'))

def test_manifest_valid_and_no_reuse(tmp_path):
    p=manifest(tmp_path)
    assert len(list(vo.manifest_rows(p,'mono')))==1 and len(list(vo.manifest_rows(p,'stereo')))==1
    with p.open('a') as f:f.write('1,1.1,l.png,r.png,1.1\n')
    with pytest.raises(ValueError):list(vo.manifest_rows(p,'stereo'))

def test_rotation_and_rest_with_initialized_map(scene,obj,monkeypatch):
    X,K,_,_=scene;d=np.random.default_rng(6).normal(size=(len(X),128)).astype(np.float32)
    centre=np.array([.4,0.,0.]);R=cv.Rodrigues(np.array([.03,.12,.01]))[0]
    points=[project(X,K),project(X,K,t=-centre),project(X,K,R,-R@centre),project(X,K,R,-R@centre)]
    frames=iter([vo.Features(p,d) for p in points]);monkeypatch.setattr(vo,'detect',lambda _:next(frames))
    e=vo.Odometry(vo.Calibration(obj,'mono'),'mono');blank=np.zeros((480,640),np.uint8)
    assert e.step(blank).state=='ANCHOR';assert e.step(blank).accepted
    c=e.C.copy();assert e.step(blank).accepted
    assert np.allclose(e.C,c,atol=1e-5) and np.allclose(e.R,R.T,atol=1e-5)
    assert e.step(blank).accepted and np.allclose(e.C,c,atol=1e-5)

def test_stereo_epipolar_and_disparity_consistency(scene,obj):
    X,K,_,_=scene;d=np.random.default_rng(2).normal(size=(len(X),128)).astype(np.float32)
    a=vo.Features(project(X,K),d);b=vo.Features(project(X,K,t=np.array([-.2,0.,0.])),d)
    cal=vo.Calibration(obj,'stereo');m,e=vo.stereo_map(a,b,cal)
    assert len(m)==len(X) and e<1e-8
    assert np.allclose(np.array([m[i] for i in range(len(X))]),X,atol=1e-6)
    assert not vo.stereo_map(b,a,cal)[0]
    shifted=vo.Features(b.points+[0,8],d)
    assert not vo.stereo_map(a,shifted,cal)[0]

def test_pnp_rejection_preserves_map_pose(scene,obj,monkeypatch):
    X,K,_,_=scene;d=np.random.default_rng(2).normal(size=(len(X),128)).astype(np.float32)
    points=[project(X,K),project(X,K,t=np.array([-.4,0.,0.])),np.random.default_rng(9).uniform([0,0],[640,480],(len(X),2))]
    frames=iter([vo.Features(p,d) for p in points]);monkeypatch.setattr(vo,'detect',lambda _:next(frames))
    e=vo.Odometry(vo.Calibration(obj,'mono'),'mono');blank=np.zeros((480,640),np.uint8)
    e.step(blank);assert e.step(blank).accepted
    R,C=e.R.copy(),e.C.copy();old=dict(e.landmarks);reference=e.ref
    assert not e.step(blank).accepted
    assert np.array_equal(e.R,R) and np.array_equal(e.C,C) and e.ref is reference
    assert all(np.array_equal(v,old[k]) for k,v in e.landmarks.items())
