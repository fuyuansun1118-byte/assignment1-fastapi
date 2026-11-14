"""Real HTTP verification for a local or containerized Assignment 4 server."""
import argparse
import hashlib
import json
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--url',default='http://127.0.0.1:8000')
p.add_argument('--output',default='Assignment4_Verification.json')
p.add_argument('--runtime',choices=['local','docker'],default='local')
a=p.parse_args()

def get(path):
    with urllib.request.urlopen(a.url+path,timeout=30) as r:return json.load(r)

for attempt in range(90):
    try:
        health=get('/health')
        assert health['ebm_ready'] and health['diffusion_ready'] and health['classifier_ready'] and health['gan_ready']
        break
    except Exception:
        if attempt==89:raise
        time.sleep(2)

out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
result={'runtime':a.runtime,'health':health,'models':{}}
for kind,steps in [('ebm',500),('diffusion',50)]:
    body=json.dumps({'num_samples':4,'steps':steps,'seed':42}).encode()
    request=urllib.request.Request(a.url+f'/{kind}/generate',data=body,
                                  headers={'Content-Type':'application/json'})
    tick=time.monotonic()
    with urllib.request.urlopen(request,timeout=180) as r:
        png=r.read();assert r.headers.get_content_type()=='image/png'
    elapsed=time.monotonic()-tick
    assert png[:8]==b'\x89PNG\r\n\x1a\n'
    dimensions=struct.unpack('>II',png[16:24]);assert dimensions==(70,70)
    with urllib.request.urlopen(request,timeout=180) as r:assert png==r.read()
    model=get(f'/{kind}/info');assert model['trained_epochs']>0
    (out.parent/f'{kind}_{a.runtime}_samples.png').write_bytes(png)
    result['models'][kind]={**model,'png_size':list(dimensions),'steps':steps,
                           'png_sha256':hashlib.sha256(png).hexdigest(),
                           'same_seed_reproducible':True,'seconds':round(elapsed,3)}
# Representative error validation over real HTTP.
request=urllib.request.Request(a.url+'/ebm/generate',data=b'{"num_samples":0}',headers={'Content-Type':'application/json'})
try:
    urllib.request.urlopen(request)
    raise AssertionError('Invalid request unexpectedly succeeded')
except urllib.error.HTTPError as e:assert e.code==422
result['invalid_request_status']=422
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
