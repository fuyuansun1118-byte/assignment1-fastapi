"""Smoke-test an already running local/Docker server using only the stdlib."""
import argparse
import hashlib
import json
import struct
import time
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:8000')
parser.add_argument('--output', default='GAN_Verification.json')
parser.add_argument('--runtime', default='local', choices=['local','docker'])
args = parser.parse_args()

def get(path):
    with urllib.request.urlopen(args.url + path, timeout=10) as r:
        return json.load(r)

for attempt in range(90):
    try:
        health = get('/health')
        assert health['gan_ready'] and health['classifier_ready']
        break
    except Exception:
        if attempt == 89: raise
        time.sleep(2)
body = json.dumps({'num_samples':16,'seed':42}).encode()
request = urllib.request.Request(args.url+'/gan/generate', data=body,
                                  headers={'Content-Type':'application/json'})
with urllib.request.urlopen(request, timeout=30) as r:
    png = r.read()
    assert r.headers.get_content_type() == 'image/png'
assert png[:8] == b'\x89PNG\r\n\x1a\n'
width,height = struct.unpack('>II', png[16:24])
assert (width,height) == (122,122)
with urllib.request.urlopen(request, timeout=30) as r:
    assert r.read() == png
info=get('/gan/info')
assert info['trained_epochs'] > 0
out=Path(args.output)
(out.parent/'gan_api_samples.png').write_bytes(png)
result={'runtime':args.runtime,'base_url':args.url,'health':health,'gan':info,
        'png_size':[width,height],'png_sha256':hashlib.sha256(png).hexdigest(),
        'same_seed_reproducible':True}
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
