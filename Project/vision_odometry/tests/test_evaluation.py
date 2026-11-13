import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import evaluate as ev

def test_global_scale_only_mono():
    rng=np.random.default_rng(4);x=rng.normal(size=(100,3));R=ev.quaternion_matrix([.1,.2,.3,.9]);y=2*x@R.T+[2,3,4]
    s,A,t=ev.align(x,y,True)
    assert np.isclose(s,2) and np.allclose(s*x@A.T+t,y)
    s,A,t=ev.align(x,y,False)
    assert s==1 and np.sqrt(np.mean(np.sum((x@A.T+t-y)**2,axis=1)))>1

def test_interpolation_gaps_and_no_extrapolation():
    t=np.array([1.,1.1,2.]);p=np.repeat(np.eye(4)[None],3,axis=0);p[1,0,3]=2
    assert ev.interpolate(t,p,.9) is None and ev.interpolate(t,p,2.1) is None
    assert ev.interpolate(t,p,1.5) is None
    assert np.isclose(ev.interpolate(t,p,1.05)[0,3],1)

def test_stationary_alignment_rejected():
    with pytest.raises(ValueError):ev.align(np.zeros((20,3)),np.zeros((20,3)),True)

def test_tf_chain_and_rotating_lever_arm(tmp_path):
    import json
    p=tmp_path/'tf.json'
    p.write_text(json.dumps([
        dict(parent='/body',child='middle',translation=[1,0,0],quaternion_xyzw=[0,0,2**-.5,2**-.5]),
        dict(parent='middle',child='camera',translation=[0,2,0],quaternion_xyzw=[0,0,0,1])]))
    T,route=ev.tf_chain(p,'body','camera');Ti,_=ev.tf_chain(p,'camera','/body')
    assert np.allclose(T[:3,3],[-1,0,0]) and np.allclose(T@Ti,np.eye(4)) and route==['body','middle','camera']
    with pytest.raises(ValueError):ev.tf_chain(p,'unknown','camera')

def test_headless_plot_ignores_inherited_notebook_backend(tmp_path):
    import csv,os,subprocess
    trajectory=tmp_path/'trajectory.csv'
    with trajectory.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['timestamp','inliers','map_points','reprojection_px','epipolar_px','accepted']);w.writerow([1,20,30,.1,.2,1])
    script=Path(ev.__file__).resolve().parent
    code="import sys;sys.path.insert(0,sys.argv[1]);import evaluate;evaluate.plot_diagnostics(sys.argv[2],[],{'meaning':'test','mode':'mono','alignment':'test'},sys.argv[3])"
    env=os.environ.copy();env['MPLBACKEND']='module://not_installed_notebook_backend'
    out=tmp_path/'plot.png'
    p=__import__('subprocess').run([sys.executable,'-c',code,str(script),str(trajectory),str(out)],env=env,capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    assert out.is_file() and out.stat().st_size>1000
