#!/usr/bin/env python3
"""Small official KITTI odometry 00 subset via HTTP ranges (NOT the 23 GB ZIP).
Data: Geiger et al., KITTI, CC BY-NC-SA 3.0; see docs/DATA.md.
Uses standard-library zipfile validation/CRC; refuses servers ignoring Range.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.request
import zipfile
import numpy as np
import cv2 as cv

BASE='https://avg-kitti.s3.eu-central-1.amazonaws.com/'

class RemoteZipFile(io.RawIOBase):
    def __init__(self,url):
        self.url=url; self.pos=0; self.transferred=0
        self.cache_start=0; self.cache=b''; self.block=1024*1024
        with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=60) as r:
            self.length=int(r.headers['Content-Length'])
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,n,whence=0):
        self.pos=n if whence==0 else self.pos+n if whence==1 else self.length+n
        return self.pos
    def read(self,n=-1):
        n=min(n if n>=0 else self.length-self.pos,self.length-self.pos)
        if n<=0:return b''
        if self.cache_start <= self.pos and self.pos+n <= self.cache_start+len(self.cache):
            offset=self.pos-self.cache_start;data=self.cache[offset:offset+n];self.pos+=n;return data
        # ZIP reads a local header, filename, extra fields, then compressed data.
        # Read-ahead makes those tiny reads share one range, avoiding hundreds
        # of latency-bound requests without downloading unrelated sequences.
        start=(self.pos//self.block)*self.block
        count=min(max(self.block,self.pos-start+n),self.length-start)
        end=start+count-1
        for attempt in range(4):
            try:
                req=urllib.request.Request(self.url+f'?range={start}-{end}',headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(req,timeout=90) as r:
                    if r.status!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):
                        raise RuntimeError('server ignored HTTP range; refusing full dataset download')
                    b=r.read(count+1)
                if len(b)!=count:raise IOError('truncated HTTP range')
                self.cache_start=start;self.cache=b;self.transferred+=count
                return self.read(n)
            except (OSError,TimeoutError):
                if attempt==3:raise
                time.sleep(1+attempt)



def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',default='data/kitti00')
    ap.add_argument('--frames',type=int,default=60)
    ap.add_argument('--start',type=int,default=0)
    a=ap.parse_args()
    if not 20<=a.frames<=500 or a.start<0:ap.error('20..500 frames; start >=0')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    hashes=[]; transferred=0
    remote=RemoteZipFile(BASE+'data_odometry_gray.zip')
    with zipfile.ZipFile(remote) as z:
        for side in (0,1):
            (out/f'image_{side}').mkdir(exist_ok=True)
            for i in range(a.start,a.start+a.frames):
                rel=f'image_{side}/{i:06d}.png'; data=z.read('dataset/sequences/00/'+rel)
                (out/rel).write_bytes(data)
                hashes.append(dict(file=rel,sha256=hashlib.sha256(data).hexdigest()))
    transferred+=remote.transferred
    for archive,member,target in [('data_odometry_calib.zip','dataset/sequences/00/calib.txt','calib.txt'),
                                  ('data_odometry_calib.zip','dataset/sequences/00/times.txt','times.txt'),
                                  ('data_odometry_poses.zip','dataset/poses/00.txt','poses.txt')]:
        rf=RemoteZipFile(BASE+archive)
        with zipfile.ZipFile(rf) as z:(out/target).write_bytes(z.read(member))
        transferred+=rf.transferred
    mats={line.split(':')[0]:np.fromstring(line.split(':')[1],sep=' ').reshape(3,4) for line in (out/'calib.txt').read_text().splitlines() if line.startswith(('P0:','P1:'))}
    P0,P1=mats['P0'],mats['P1'];K=P0[:,:3]
    if not np.allclose(K,P1[:,:3]) or abs(P0[0,3])>1e-8:raise ValueError('unsupported KITTI projection')
    im=cv.imread(str(out/f'image_0/{a.start:06d}.png'),0);h,w=im.shape
    cal=dict(width=w,height=h,K_left=K.tolist(),K_right=K.tolist(),D_left=[0.]*5,D_right=[0.]*5,
             R_right_left=np.eye(3).tolist(),T_right_left=[P1[0,3]/K[0,0],0.,0.],translation_unit='m',already_rectified=True,
             provenance='Official KITTI odometry sequence 00 calib.txt P0/P1; rectified grayscale cameras')
    (out/'calibration.json').write_text(json.dumps(cal,indent=2)+'\n')
    times=(out/'times.txt').read_text().splitlines()
    with (out/'frames.csv').open('w',newline='') as f:
        wr=csv.writer(f);wr.writerow(['index','timestamp','left','right','right_timestamp'])
        for i in range(a.start,a.start+a.frames):wr.writerow([i,times[i],f'image_0/{i:06d}.png',f'image_1/{i:06d}.png',times[i]])
    poses=np.loadtxt(out/'poses.txt').reshape(-1,3,4)
    with (out/'reference.csv').open('w',newline='') as f:
        wr=csv.writer(f);wr.writerow(['timestamp','x','y','z']+[f'r{i}{j}' for i in range(3) for j in range(3)])
        for i in range(a.start,a.start+a.frames):wr.writerow([times[i],*poses[i,:,3],*poses[i,:,:3].ravel()])
    (out/'provenance.json').write_text(json.dumps(dict(source=BASE,sequence='00',start=a.start,frames=a.frames,downloaded_bytes=transferred,license='CC-BY-NC-SA-3.0',hashes=hashes),indent=2)+'\n')
    print(f'KITTI 00: {a.frames} real pairs; transferred {transferred/1e6:.1f} MB')

if __name__=='__main__':main()
