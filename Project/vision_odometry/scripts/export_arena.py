#!/usr/bin/env python3
"""Export real rectified IR pairs + CameraInfo + raw IMUs from ROS2 MCAP.
No ROS installation is needed: pip install mcap mcap-ros2-support.
No images are synthesized; pair by integer Header stamp, one-to-one.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import cv2 as cv
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

BASE = '/kalibr/rgbd/'
LEFT = BASE+'infra1/image_rect_raw/compressed'
RIGHT = BASE+'infra2/image_rect_raw/compressed'
INFOS = [BASE+'infra1/camera_info', BASE+'infra2/camera_info']
IMUS = [BASE+'imu', BASE+'gyro/sample', BASE+'accel/sample', '/kalibr/livox/imu']


def ns(msg):
    return int(msg.header.stamp.sec)*10**9 + int(msg.header.stamp.nanosec)


def stamp(n):
    return f'{n//10**9}.{n%10**9:09d}'


def save_csv(path, fields, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(fields); w.writerows(rows)


def extract_fast_csv(path, out, start_ns, end_ns):
    """PlotJuggler export: use ORIGINAL odometry header, never row time."""
    prefix = '/kalibr/fast_lio/odometry/'
    pose = prefix+'pose/pose/'
    fields = ['timestamp','x','y','z','qx','qy','qz','qw','frame_id','child_frame_id']
    rows, seen = [], set()
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            sec, nano = r.get(prefix+'header/stamp/sec'), r.get(prefix+'header/stamp/nanosec')
            if not sec or not nano:
                continue
            n = int(sec)*10**9+int(nano)
            if n in seen or not start_ns <= n <= end_ns:
                continue
            vals = [r.get(pose+'position/'+c,'') for c in 'xyz'] + [r.get(pose+'orientation/'+c,'') for c in ['x','y','z','w']]
            if not all(vals):
                continue
            seen.add(n)
            rows.append([stamp(n), *map(float, vals), r.get(prefix+'header/frame_id',''), r.get(prefix+'child_frame_id','')])
    rows.sort(key=lambda x: float(x[0]))
    save_csv(out/'reference_fastlio.csv', fields, rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bag', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--fastlio-csv')
    ap.add_argument('--start', type=float, default=20., help='seconds from first IR image')
    ap.add_argument('--duration', type=float, default=20.)
    ap.add_argument('--stride', type=int, default=6)
    a = ap.parse_args()
    if a.start < 0 or a.duration <= 0 or a.stride < 1:
        ap.error('invalid extraction interval/stride')
    paths = sorted(Path(a.bag).glob('*.mcap')) if Path(a.bag).is_dir() else [Path(a.bag)]
    if not paths:
        raise ValueError('no MCAP files')
    out = Path(a.output)
    if (out/'frames.csv').exists():
        raise ValueError('output already contains a dataset; choose a new directory')
    for side in ['left','right']:
        (out/side).mkdir(parents=True, exist_ok=True)
    info, images, tf, imu_rows = {}, {LEFT:{}, RIGHT:{}}, {}, []
    first, start, end = None, None, None
    total_left = 0
    selected = set()
    topics = [LEFT, RIGHT, *INFOS, *IMUS, '/tf_static']
    for path in paths:
        with path.open('rb') as f:
            r = make_reader(f, decoder_factories=[DecoderFactory()])
            for _, ch, msg, d in r.iter_decoded_messages(topics=topics):
                topic = ch.topic
                if topic in INFOS and topic not in info:
                    info[topic] = {'width':d.width,'height':d.height,'K':list(d.k),'D':list(d.d),'R':list(d.r),'P':list(d.p),'frame_id':d.header.frame_id,'distortion_model':d.distortion_model}
                elif topic == '/tf_static':
                    for x in d.transforms:
                        tr, q = x.transform.translation, x.transform.rotation
                        tf[(x.header.frame_id,x.child_frame_id)] = {'parent':x.header.frame_id,'child':x.child_frame_id,'translation':[tr.x,tr.y,tr.z],'quaternion_xyzw':[q.x,q.y,q.z,q.w]}
                elif topic == LEFT:
                    n = ns(d)
                    if first is None:
                        first = n; start = n+int(a.start*1e9); end = start+int(a.duration*1e9)
                    if start <= n < end:
                        if total_left % a.stride == 0:
                            selected.add(n); images[LEFT][n] = bytes(d.data)
                        total_left += 1
                elif topic == RIGHT:
                    # Right may arrive before left; retain only the selected time window.
                    n = ns(d)
                    if start is not None and start <= n < end:
                        images[RIGHT][n] = bytes(d.data)
                elif topic in IMUS:
                    n = ns(d)
                    if start is not None and start <= n < end:
                        g, acc = d.angular_velocity, d.linear_acceleration
                        imu_rows.append([stamp(n),topic,d.header.frame_id,g.x,g.y,g.z,acc.x,acc.y,acc.z])
    if len(info) != 2 or not selected:
        raise ValueError('missing IR images or CameraInfo')
    l, r = info[INFOS[0]], info[INFOS[1]]
    K = np.array(l['P']).reshape(3,4)[:,:3]
    Kr = np.array(r['P']).reshape(3,4)[:,:3]
    # These topics contain rectified pixels. P is their projection calibration;
    # do not apply an unrelated raw-image Kalibr distortion model a second time.
    if not np.allclose(K,Kr) or not np.allclose(np.array(l['R']).reshape(3,3),np.eye(3)) or not np.allclose(np.array(r['R']).reshape(3,3),np.eye(3)):
        raise ValueError('exporter requires common, rectified projection geometry')
    if abs(l['P'][3]) > 1e-8 or abs(r['P'][7]) > 1e-8:
        raise ValueError('unsupported rectified projection')
    tx = (r['P'][3]-l['P'][3])/K[0,0]
    if tx >= 0:
        raise ValueError('invalid stereo baseline in CameraInfo.P')
    cal = dict(width=l['width'],height=l['height'],K_left=K.tolist(),K_right=Kr.tolist(),D_left=[0.]*5,D_right=[0.]*5,
               R_right_left=np.eye(3).tolist(),T_right_left=[tx,0.,0.],translation_unit='m',already_rectified=True,
               provenance='Recorded ROS CameraInfo.P for rectified IR streams; factory/driver calibration, not a new physical calibration',
               convention='X_right = R_right_left X_left + T_right_left; optical x right y down z forward')
    rows, hashes = [], []
    for n in sorted(selected):
        if n not in images[RIGHT]:
            continue
        idx = len(rows)
        paths_row=[]
        decoded=[]
        for topic,side in [(LEFT,'left'),(RIGHT,'right')]:
            data=images[topic][n]; im=cv.imdecode(np.frombuffer(data,np.uint8),cv.IMREAD_GRAYSCALE)
            if im is None or im.shape != (l['height'],l['width']):
                raise ValueError('corrupt/mismatched image payload')
            decoded.append(im)
            name=f'{side}/{idx:06d}.png'; cv.imwrite(str(out/name),im); paths_row.append(name)
            hashes.append({'file':name,'sha256':hashlib.sha256((out/name).read_bytes()).hexdigest()})
        if np.array_equal(*decoded):
            raise ValueError('identical left/right images; refusing fake stereo')
        rows.append([idx,stamp(n),paths_row[0],paths_row[1],stamp(n)])
    if not rows:
        raise ValueError('no exact timestamp stereo pairs')
    save_csv(out/'frames.csv',['index','timestamp','left','right','right_timestamp'],rows)
    save_csv(out/'imu.csv',['timestamp','topic','frame_id','gx','gy','gz','ax','ay','az'],imu_rows)
    (out/'calibration.json').write_text(json.dumps(cal,indent=2)+'\n')
    (out/'camera_info_recorded.json').write_text(json.dumps(info,indent=2)+'\n')
    (out/'tf_static_recorded.json').write_text(json.dumps(list(tf.values()),indent=2)+'\n')
    nref = extract_fast_csv(a.fastlio_csv,out,start,end) if a.fastlio_csv else 0
    provenance=dict(bag_name=Path(a.bag).name,files=[{'name':p.name,'bytes':p.stat().st_size} for p in paths],
                    start_seconds_from_first_image=a.start,duration_seconds=a.duration,stride=a.stride,
                    pairs=len(rows),unpaired_selected_left=len(selected)-len(rows),pairing='exact integer header nanoseconds; no reuse',
                    max_stereo_delta_ns=0,baseline_m=-tx,reference_samples=nref,reference='FAST-LIO estimate, NOT ground truth',
                    imu='Preserved independently; gyro rad/s, RealSense accel m/s^2; Livox raw acceleration may be g. Not input to visual estimator.',
                    hashes=hashes)
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({k:v for k,v in provenance.items() if k!='hashes'},indent=2))

if __name__=='__main__':
    main()
