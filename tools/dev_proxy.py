#!/usr/bin/env python3
"""로컬 개발용 서울 실시간 도시데이터 중계 서버 (netlify/functions/seoul.js 와 같은 동작).
실행:  py -3 tools/dev_proxy.py            → http://127.0.0.1:8002/seoul?area=광화문·덕수궁&key=인증키
환경변수 SEOUL_API_KEY 또는 저장소 루트의 .env 파일(SEOUL_API_KEY=...)로 기본 키 지정. 키가 없으면 'sample'(광화문·덕수궁만 동작).
"""
import json, os, sys, urllib.parse, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE = "http://openapi.seoul.go.kr:8088"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8002

# 저장소 루트의 .env (git 에 올라가지 않음) 에서 SEOUL_API_KEY=... 를 읽는다
_env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(_env):
    for _line in open(_env, encoding="utf-8"):
        if "=" in _line and not _line.strip().startswith("#"):
            _k, _v = _line.strip().split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())


def get_json(url):
    with urllib.request.urlopen(url, timeout=8) as r:
        return json.load(r)


def simplify(row, source):
    keys = ["AREA_NM", "AREA_CD", "AREA_CONGEST_LVL", "AREA_CONGEST_MSG", "AREA_PPLTN_MIN", "AREA_PPLTN_MAX", "PPLTN_TIME"]
    out = {k: row.get(k) for k in keys}
    out["source"] = source
    return out


def lookup(area, key):
    enc = urllib.parse.quote(area)
    js = get_json(f"{BASE}/{urllib.parse.quote(key)}/json/citydata_ppltn/1/5/{enc}")
    rows = js.get("SeoulRtd.citydata_ppltn") or []
    if rows:
        return simplify(rows[0], "citydata_ppltn"), None
    js = get_json(f"{BASE}/{urllib.parse.quote(key)}/json/citydata/1/5/{enc}")
    cd = js.get("CITYDATA") or {}
    rows = cd.get("LIVE_PPLTN_STTS") or []
    if rows:
        row = dict(rows[0])
        row.setdefault("AREA_NM", cd.get("AREA_NM"))
        row.setdefault("AREA_CD", cd.get("AREA_CD"))
        return simplify(row, "citydata"), None
    msg = (js.get("RESULT") or {}).get("MESSAGE") or json.dumps(js, ensure_ascii=False)[:120]
    return None, msg


class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path.rstrip("/") not in ("/seoul", ""):
            return self._send(404, {"error": "not found"})
        q = urllib.parse.parse_qs(u.query)
        area = (q.get("area") or [""])[0].strip()
        key = (q.get("key") or [os.environ.get("SEOUL_API_KEY", "sample")])[0].strip() or "sample"
        if not area:
            return self._send(400, {"error": "area 가 필요합니다"})
        try:
            row, err = lookup(area, key)
        except Exception as e:
            return self._send(502, {"error": str(e), "area": area})
        if row is None:
            return self._send(502, {"error": err, "area": area})
        if row.get("AREA_NM") and row["AREA_NM"] != area:
            # sample 키는 어떤 장소를 물어도 광화문·덕수궁을 돌려준다 → 다른 장소로 쓰면 안 됨
            return self._send(404, {"error": f"요청한 장소({area})와 다른 응답({row['AREA_NM']}). 샘플 키는 광화문·덕수궁만 제공합니다", "area": area})
        self._send(200, row)

    def log_message(self, fmt, *a):
        sys.stderr.write("[seoul-proxy] " + (fmt % a) + chr(10))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"서울 도시데이터 중계 서버: http://127.0.0.1:{PORT}/seoul?area=광화문·덕수궁", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
