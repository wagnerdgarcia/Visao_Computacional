#!/usr/bin/env python3
"""Reproduce legacy failures from immutable base commit, without GUI or checkout changes.
Run from any directory in this Git repository; C++ compiler/OpenCV required.
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
BASE='3d347e2'

def main():
    report={'base_commit':subprocess.check_output(['git','rev-parse',BASE],cwd=REPO,text=True).strip(),'checks':{}}
    with tempfile.TemporaryDirectory(prefix='vo-before-') as temporary:
        p=Path(temporary);(p/'src').mkdir();(p/'include').mkdir()
        for rel in ['Makefile','include/mono_odometry.h','src/mono_odometry.cpp']:
            content=subprocess.check_output(['git','show',f'{BASE}:Project/vision_odometry/{rel}'],cwd=REPO)
            (p/rel).write_bytes(content)
        for key,cmd in [('build',['make','-B']),('clean',['make','clean'])]:
            r=subprocess.run(cmd,cwd=p,capture_output=True,text=True)
            report['checks'][key]={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
        (p/'probe.cpp').write_text('''#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <filesystem>
#include <numeric>
#define private public
#define main old_main
#include "src/mono_odometry.cpp"
#undef main
#undef private
int main(){
 MonoOdometry v("unused"); auto [d,x,y]=v.calculateDistances({},{},{});
 std::cout<<"empty_distances="<<d<<","<<x<<","<<y<<" finite="<<std::isfinite(d)<<"\\n";
 std::vector<cv::Point2f>a{{1,2},{3,4}},b{{5,6},{7,8}};
 cv::Mat blank=cv::Mat::zeros(480,640,CV_8U);v.trackFeatures(blank,blank,a,b);
 std::cout<<"stale_output_sizes_after_blank="<<a.size()<<","<<b.size()<<"\\n";
 cv::Mat aa(3,128,CV_32F,cv::Scalar(0)),bb(1,128,CV_32F,cv::Scalar(1));std::vector<std::vector<cv::DMatch>>m;
 try{cv::FlannBasedMatcher().knnMatch(aa,bb,m,2);std::cout<<"neighbors="<<m.at(0).size()<<"\\n";}
 catch(cv::Exception&e){std::cout<<"single_train_exception="<<e.what()<<"\\n";}
 cv::Mat Rold=(cv::Mat_<double>(3,3)<<0,-1,0,1,0,0,0,0,1),Rrel=cv::Mat::eye(3,3,CV_64F),t=(cv::Mat_<double>(3,1)<<1,0,0);
 cv::Mat wrong=Rrel*Rold*t,correct=-Rold*Rrel.t()*t;
 std::cout<<"legacy_displacement="<<wrong.t()<<" expected="<<correct.t()<<"\\n";
}''')
        flags=subprocess.check_output(['pkg-config','--cflags','--libs','opencv4'],text=True).split()
        subprocess.run(['g++','-std=c++17','-Iinclude','probe.cpp','-o','probe',*flags],cwd=p,check=True,capture_output=True)
        r=subprocess.run([str(p/'probe')],capture_output=True,text=True,check=True)
        report['checks']['probe']={'stdout':r.stdout,'stderr':r.stderr}
    # Empirical retained counts; same SIFT descriptors, old gates versus ratio+mutual.
    import cv2 as cv
    import numpy as np
    cv.setNumThreads(1)
    import sys
    sys.path.insert(0,str(ROOT/'python'));import vo
    cap=cv.VideoCapture(str(ROOT/'video/video_simulacao.mp4'))
    counts=[]
    for i,j in [(0,1),(30,31),(150,151),(150,165),(600,601)]:
        fs=[]
        for n in (i,j):
            cap.set(cv.CAP_PROP_POS_FRAMES,n);ok,im=cap.read()
            if not ok:raise RuntimeError('cannot decode inherited video')
            kp,d=cv.SIFT_create().detectAndCompute(cv.cvtColor(im,cv.COLOR_BGR2GRAY),None)
            fs.append(vo.Features(np.array([k.pt for k in kp]),d))
        a,b=fs
        matches=cv.BFMatcher().knnMatch(a.descriptors,b.descriptors,k=2)
        strict=[m[0] for m in matches if len(m)>=2 and m[0].distance<.4*m[1].distance and m[0].distance<30 and np.linalg.norm(a.points[m[0].queryIdx]-b.points[m[0].trainIdx])<10]
        counts.append({'previous_frame':i,'current_frame':j,'legacy_gates':len(strict),'ratio_mutual_no_displacement_cap':len(vo.match(a,b))})
    cap.release();report['checks']['video_match_counts']=counts
    dest=ROOT/'docs/evidence/reproduction-before.json';dest.write_text(json.dumps(report,indent=2)+'\n');print(dest)
if __name__=='__main__':main()
