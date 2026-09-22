"""Capture original synthetic HTTP bytes; no DTO conversion or normalization."""
import hashlib
import json
import os
from pathlib import Path
from .seed import PROMOTION


def capture(case, request_id, request, response):
    directory = Path(os.environ['RCG_OUTPUT']) / 'responses'
    directory.mkdir(exist_ok=True)
    stem = case + '-' + request_id
    body = response.content
    (directory/(stem+'.body')).write_bytes(body)
    metadata = dict(case_id=case, request_id=request_id, method='POST',
                    path='/decision/v1/promotions/'+PROMOTION+'/runs', request=request,
                    status=response.status_code, content_type=response.headers.get('content-type'),
                    body_file=stem+'.body', body_sha256=hashlib.sha256(body).hexdigest())
    (directory/(stem+'.json')).write_text(json.dumps(metadata, sort_keys=True, indent=2)+'\n')
    return metadata
