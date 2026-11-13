"""Input failures and transactional output behavior in both actual executables."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import cv2 as cv
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(params=['python','cpp'])
def executable(request):
    if request.param=='cpp':
        binary=ROOT/'build/vision_odometry'
        if not binary.exists():pytest.skip('compile C++ before native CLI integration tests')
        return [str(binary)]
    return [sys.executable,str(ROOT/'python/vo.py')]

@pytest.mark.parametrize('bad',['missing_right','desync','duplicate','resolution','missing_calibration','nonmonotonic'])
def test_rejected_inputs(executable,tmp_path,bad):
    image=np.zeros((480,640),np.uint8);cv.imwrite(str(tmp_path/'left.png'),image);cv.imwrite(str(tmp_path/'right.png'),image)
    obj=json.loads((ROOT/'config/synthetic.json').read_text())
    row=[0,1.,'left.png','right.png',1.]
    if bad=='missing_right':row[3]='absent.png'
    if bad=='desync':row[4]=1.1
    if bad=='duplicate':row[3]='left.png'
    if bad=='resolution':obj['width']=641
    if bad=='missing_calibration':obj.pop('K_right')
    cal=tmp_path/'cal.json';cal.write_text(json.dumps(obj))
    manifest=tmp_path/'frames.csv'
    with manifest.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['index','timestamp','left','right','right_timestamp']);w.writerow(row)
        if bad=='nonmonotonic':w.writerow([1,.9,'left.png','right.png',.9])
    out=tmp_path/'out.csv';p=subprocess.run(executable+['--mode','stereo','--calibration',str(cal),'--manifest',str(manifest),'--output',str(out)],capture_output=True,text=True)
    assert p.returncode!=0 and 'error' in p.stderr.lower()
    if out.exists():assert len(list(csv.DictReader(out.open())))==0

def test_blank_sequence_reinitializes(executable,tmp_path):
    image=np.zeros((480,640),np.uint8);manifest=tmp_path/'frames.csv'
    with manifest.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['index','timestamp','left'])
        for i in range(4):
            name=f'{i}.png';cv.imwrite(str(tmp_path/name),image);w.writerow([i,i/10,name])
    out=tmp_path/'out.csv';p=subprocess.run(executable+['--mode','mono','--calibration',str(ROOT/'config/synthetic.json'),'--manifest',str(manifest),'--output',str(out)],capture_output=True,text=True)
    assert p.returncode==0
    rows=list(csv.DictReader(out.open()));assert len(rows)==4 and rows[2]['state']=='REINITIALIZING'
    assert all(r['accepted']=='0' and r['pose_valid']=='0' and float(r['x'])==0 for r in rows)
