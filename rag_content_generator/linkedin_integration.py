import json
import os
import time
import secrets
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError

TOKEN_FILE = Path('.linkedin_token.json')


def _parse_env_file(path: Path = Path('.env')):
    if not path.exists():
        return
    for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and v and not os.getenv(k):
            os.environ[k] = v


def get_config():
    _parse_env_file()
    return {
        'client_id': os.getenv('LINKEDIN_CLIENT_ID', ''),
        'client_secret': os.getenv('LINKEDIN_CLIENT_SECRET', ''),
        'redirect_uri': os.getenv('LINKEDIN_REDIRECT_URI', 'http://localhost:8501'),
    }


def config_ready(cfg):
    return all([cfg.get('client_id'), cfg.get('client_secret'), cfg.get('redirect_uri')])


def build_auth_url(cfg, state):
    params = {
        'response_type': 'code',
        'client_id': cfg['client_id'],
        'redirect_uri': cfg['redirect_uri'],
        'state': state,
        'scope': 'openid profile w_member_social',
    }
    return 'https://www.linkedin.com/oauth/v2/authorization?' + urlencode(params)


def _http_post_json(url, data, headers=None):
    body = urlencode(data).encode('utf-8')
    h = {'Content-Type': 'application/x-www-form-urlencoded'}
    if headers:
        h.update(headers)
    req = Request(url, data=body, headers=h, method='POST')
    with urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode('utf-8'))


def _http_get_json(url, headers=None):
    req = Request(url, headers=headers or {}, method='GET')
    with urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode('utf-8'))


def exchange_code_for_token(cfg, code):
    payload = {
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri': cfg['redirect_uri'],
        'client_id': cfg['client_id'],
        'client_secret': cfg['client_secret'],
    }
    tok = _http_post_json('https://www.linkedin.com/oauth/v2/accessToken', payload)
    tok['obtained_at'] = int(time.time())
    save_token(tok)
    return tok


def save_token(token):
    TOKEN_FILE.write_text(json.dumps(token, indent=2), encoding='utf-8')


def load_token():
    if not TOKEN_FILE.exists():
        return None
    try:
        return json.loads(TOKEN_FILE.read_text(encoding='utf-8'))
    except Exception:
        return None


def clear_token():
    if TOKEN_FILE.exists():
        TOKEN_FILE.unlink()


def get_access_token():
    tok = load_token()
    if not tok:
        return None
    if 'access_token' not in tok:
        return None
    return tok['access_token']


def get_person_urn(access_token):
    # OIDC endpoint provides stable subject id for person
    data = _http_get_json(
        'https://api.linkedin.com/v2/userinfo',
        headers={'Authorization': f'Bearer {access_token}'}
    )
    sub = data.get('sub')
    if not sub:
        raise RuntimeError('Could not fetch LinkedIn user id (sub).')
    return f'urn:li:person:{sub}'


def post_text(access_token, text):
    # Use stable v2 UGC endpoint for compatibility
    author = get_person_urn(access_token)
    payload = {
        "author": author,
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {"text": text},
                "shareMediaCategory": "NONE"
            }
        },
        "visibility": {
            "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
        }
    }

    req = Request(
        'https://api.linkedin.com/v2/ugcPosts',
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json',
            'X-Restli-Protocol-Version': '2.0.0',
        },
        method='POST'
    )
    try:
        with urlopen(req, timeout=30) as r:
            body = r.read().decode('utf-8', errors='ignore')
            return {'ok': True, 'status': r.status, 'body': body}
    except HTTPError as e:
        detail = e.read().decode('utf-8', errors='ignore')
        return {'ok': False, 'status': e.code, 'body': detail}


def new_state():
    return secrets.token_urlsafe(16)
