"""Remote ZIP random access and explicit refusal of a full HTTP response."""
import io
from pathlib import Path
import re
import sys
import zipfile
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import download_kitti as dk

class Response:
    def __init__(self,data,status,headers):self.data=data;self.status=status;self.headers=headers
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def read(self,n):return self.data[:n]

def test_remote_zip_random_access_and_read_ahead(monkeypatch):
    data=io.BytesIO()
    with zipfile.ZipFile(data,'w',zipfile.ZIP_DEFLATED) as z:
        for i in range(8):z.writestr(f'{i}.txt',(f'file {i} contents '*100).encode())
    payload=data.getvalue();requests=[]
    def urlopen(req,timeout):
        if req.get_method()=='HEAD':return Response(b'',200,{'Content-Length':str(len(payload))})
        requests.append(req)
        a,b=map(int,re.fullmatch(r'bytes=(\d+)-(\d+)',req.headers['Range']).groups())
        return Response(payload[a:b+1],206,{'Content-Range':f'bytes {a}-{b}/{len(payload)}'})
    monkeypatch.setattr(dk.urllib.request,'urlopen',urlopen)
    remote=dk.RemoteZipFile('https://example.test/dataset.zip')
    with zipfile.ZipFile(remote) as z:
        for i in [7,0,6,2,3]:assert z.read(f'{i}.txt')==(f'file {i} contents '*100).encode()
    assert len(requests)==1 and remote.transferred==len(payload)

def test_refuse_server_ignoring_range(monkeypatch):
    def urlopen(req,timeout):
        return Response(b'x'*100,200,{'Content-Length':'100'})
    monkeypatch.setattr(dk.urllib.request,'urlopen',urlopen)
    with pytest.raises(RuntimeError,match='ignored HTTP range'):dk.RemoteZipFile('https://example.test/big.zip').read(2)
