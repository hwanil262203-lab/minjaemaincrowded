// Netlify Function: 서울시 실시간 도시데이터(인구·혼잡도) 중계
// 브라우저는 openapi.seoul.go.kr 를 직접 부를 수 없다 (HTTP 전용 + CORS 없음).
// GET /.netlify/functions/seoul?area=광화문·덕수궁[&key=인증키]
// 인증키는 쿼리 key 또는 환경변수 SEOUL_API_KEY. 둘 다 없으면 'sample'(광화문·덕수궁만 동작).
// 응답: { AREA_NM, AREA_CD, AREA_CONGEST_LVL, AREA_CONGEST_MSG, AREA_PPLTN_MIN, AREA_PPLTN_MAX, PPLTN_TIME, source }
const BASE = 'http://openapi.seoul.go.kr:8088';
const CORS = { 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'GET, OPTIONS', 'Access-Control-Allow-Headers': 'Content-Type' };

function simplify(row, source) {
  if (!row) return null;
  return {
    AREA_NM: row.AREA_NM, AREA_CD: row.AREA_CD, AREA_CONGEST_LVL: row.AREA_CONGEST_LVL, AREA_CONGEST_MSG: row.AREA_CONGEST_MSG,
    AREA_PPLTN_MIN: row.AREA_PPLTN_MIN, AREA_PPLTN_MAX: row.AREA_PPLTN_MAX, PPLTN_TIME: row.PPLTN_TIME, source,
  };
}

async function getJson(url) {
  const r = await fetch(url, { signal: AbortSignal.timeout(8000) });
  const text = await r.text();
  try { return JSON.parse(text); } catch (_) { throw new Error('JSON 아님: ' + text.slice(0, 80)); }
}

exports.handler = async (event) => {
  if (event.httpMethod === 'OPTIONS') return { statusCode: 204, headers: CORS, body: '' };
  const q = event.queryStringParameters || {};
  const area = (q.area || '').trim();
  const key = (q.key || process.env.SEOUL_API_KEY || 'sample').trim();
  if (!area) return { statusCode: 400, headers: CORS, body: JSON.stringify({ error: 'area 가 필요합니다' }) };
  const enc = encodeURIComponent(area);
  try {
    // 1) 인구 전용(가벼움)
    let js = await getJson(`${BASE}/${encodeURIComponent(key)}/json/citydata_ppltn/1/5/${enc}`);
    let row = (js['SeoulRtd.citydata_ppltn'] || [])[0];
    let source = 'citydata_ppltn';
    // 2) 실패하면 전체 도시데이터에서 인구 부분만
    if (!row) {
      js = await getJson(`${BASE}/${encodeURIComponent(key)}/json/citydata/1/5/${enc}`);
      const cd = js.CITYDATA || {};
      row = (cd.LIVE_PPLTN_STTS || [])[0];
      if (row) row = { AREA_NM: cd.AREA_NM, AREA_CD: cd.AREA_CD, ...row };
      source = 'citydata';
    }
    if (!row) {
      const msg = (js.RESULT && js.RESULT.MESSAGE) || JSON.stringify(js).slice(0, 120) || '데이터 없음';
      return { statusCode: 502, headers: CORS, body: JSON.stringify({ error: msg, area }) };
    }
    if (row.AREA_NM && row.AREA_NM !== area) {
      // sample 키는 어떤 장소를 물어도 광화문·덕수궁을 돌려준다 → 다른 장소로 쓰면 안 됨
      return { statusCode: 404, headers: CORS, body: JSON.stringify({ error: `요청한 장소(${area})와 다른 응답(${row.AREA_NM}). 샘플 키는 광화문·덕수궁만 제공합니다`, area }) };
    }
    return { statusCode: 200, headers: { ...CORS, 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'public, max-age=120' }, body: JSON.stringify(simplify(row, source)) };
  } catch (e) {
    return { statusCode: 502, headers: CORS, body: JSON.stringify({ error: e.message, area }) };
  }
};
