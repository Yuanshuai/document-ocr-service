import os
import requests
from urllib.parse import urljoin

# ====== 对象存储模式选择 ======
# STORAGE_MODE: "proxy" (默认，通过HTTP代理) 或 "minio" (直连MinIO)
# MinIO直连需要设置: MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET
STORAGE_MODE = os.environ.get('STORAGE_MODE', 'proxy').strip().lower()

if STORAGE_MODE == 'minio':
    from minio import Minio
    from minio.error import S3Error

    _minio_client = Minio(
        os.environ['MINIO_ENDPOINT'],
        access_key=os.environ['MINIO_ACCESS_KEY'],
        secret_key=os.environ['MINIO_SECRET_KEY'],
        secure=os.environ.get('MINIO_SECURE', 'false').lower() == 'true',
    )
    _minio_bucket = os.environ['MINIO_BUCKET']

    def getObject(key):
        try:
            resp = _minio_client.get_object(_minio_bucket, key)
            return resp.read()
        finally:
            if resp:
                resp.close()
                resp.release_conn()

    def putObject(key, filename):
        import os as _os
        _minio_client.fput_object(_minio_bucket, key, filename)
        return {'code': 200}

else:
    # proxy mode (default)
    S3_PROXY_SERVER = os.environ.get('S3_PROXY_SERVER', 'http://127.0.0.1:30230').strip()
    if not S3_PROXY_SERVER:
        print('Missing S3_PROXY_SERVER env var')
        os.abort()

    def getObject(key):
        resp = requests.get(urljoin(S3_PROXY_SERVER, 'getObject'), params={'key': key})
        resp.raise_for_status()
        return resp.content

    def putObject(key, filename):
        files = {
            'file': (os.path.basename(filename), open(filename, 'rb'), 'application/octet-stream')
        }
        resp = requests.post(urljoin(S3_PROXY_SERVER, 'putObject'), params={'key': key}, files=files)
        resp.raise_for_status()
        respData = resp.json()
        print(respData)
        if respData.get('code') != 200:
            msg = f'{respData["msg"] or "Upload failed"} for url: {resp.url}'
            raise requests.HTTPError(msg, resp)
