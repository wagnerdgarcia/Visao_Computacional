#!/usr/bin/env python3
"""Run C++/Python on identical manifests; evaluate only after estimation."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--datasets',nargs='+',default=['arena1','arena2','arena3','kitti00'])
    ap.add_argument('--jobs',type=int,default=2);ap.add_argument('--legacy-video',action='store_true');a=ap.parse_args()
    out=ROOT/'results';out.mkdir(exist_ok=True);commands=[]
    for ds in a.datasets+(['legacy_video'] if a.legacy_video else []):
        for lang in ['python','cpp']:
            for mode in ['mono','stereo']:
                if ds=='legacy_video' and mode=='stereo':continue
                cmd=[sys.executable,'python/vo.py'] if lang=='python' else ['build/vision_odometry']
                cmd+=['--mode',mode,'--output',f'results/{ds}_{lang}_{mode}.csv']
                if ds=='legacy_video':cmd+=['--video','video/video_simulacao.mp4','--calibration','config/legacy_video_UNVERIFIED.json','--stride','3']
                else:cmd+=['--manifest',f'data/{ds}/frames.csv','--calibration',f'data/{ds}/calibration.json']
                commands.append((ds,lang,mode,cmd))
    def execute(item):
        ds,lang,mode,cmd=item;start=time.monotonic();p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
        result=dict(dataset=ds,implementation=lang,mode=mode,command=cmd,returncode=p.returncode,elapsed_seconds=time.monotonic()-start,stdout=p.stdout,stderr=p.stderr)
        (out/f'{ds}_{lang}_{mode}_run.json').write_text(json.dumps(result,indent=2)+'\n');return result
    records=[]
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        for f in as_completed([pool.submit(execute,c) for c in commands]):
            r=f.result();records.append(r);print(f"{r['dataset']} {r['implementation']} {r['mode']}: exit={r['returncode']} {r['elapsed_seconds']:.1f}s",flush=True)
    (out/'runs.json').write_text(json.dumps(records,indent=2)+'\n')
    if any(r['returncode'] for r in records):raise SystemExit('at least one real run failed')
    import evaluate
    for ds in a.datasets:
        for lang in ['python','cpp']:
            for mode in ['mono','stereo']:
                stem=out/f'{ds}_{lang}_{mode}';directory=ROOT/'data'/ds;arena=ds.startswith('arena')
                summary,points=evaluate.run(stem.with_suffix('.csv'),directory/('reference_fastlio.csv' if arena else 'reference.csv'),mode,'fastlio' if arena else 'ground_truth',directory/'tf_static_recorded.json' if arena else None)
                stem.with_name(stem.name+'_evaluation').with_suffix('.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
                evaluate.plot_diagnostics(stem.with_suffix('.csv'),points,summary,stem.with_suffix('.png'))
    print('Evaluation complete. Arena metrics are disagreement, not ground-truth accuracy.',flush=True)
if __name__=='__main__':main()
