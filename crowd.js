/*
  crowd.js — 인파 밀집도 추정 모듈
  두 겹으로 동작한다.
  1) 전국 공통(키 불필요): OpenStreetMap 의 사람이 모이는 지점(역·정류장·시장·학교·상점 등)을 종류별 가중치로 합산하고,
     요일·시간대(출퇴근·점심·하교·주말 오후·심야) 계수를 곱해 경로의 '인파 지수'를 만든다.
  2) 서울(선택, API 키 + 중계 서버 필요): 서울시 실시간 도시데이터(citydata_ppltn) 의 장소별 혼잡도
     (여유/보통/약간 붐빔/붐빔) 를 경로 근처 장소에 적용한다. 브라우저에서 직접 부를 수 없어(HTTP·CORS)
     netlify/functions/seoul.js 또는 tools/dev_proxy.py 를 거친다.
*/
const Crowd = (() => {
  const MOCK = new URLSearchParams(location.search).get('mock') === '1';

  // ---------- 1) OSM 지점 가중치 ----------
  const RULES = [
    [(t) => t.railway === 'station' || t.railway === 'subway_entrance' || t.public_transport === 'station', 3.0, '역'],
    [(t) => t.amenity === 'bus_station' || t.highway === 'bus_stop' || t.public_transport === 'platform' || t.public_transport === 'stop_position', 1.2, '정류장'],
    [(t) => t.amenity === 'marketplace' || t.shop === 'mall' || t.shop === 'department_store' || t.shop === 'supermarket', 2.5, '시장·대형매장'],
    [(t) => t.amenity === 'school' || t.amenity === 'university' || t.amenity === 'college', 2.0, '학교'],
    [(t) => ['restaurant', 'cafe', 'fast_food', 'bar', 'pub', 'cinema', 'theatre'].includes(t.amenity), 1.0, '음식점·문화'],
    [(t) => !!t.shop, 0.8, '상점'],
    [(t) => t.leisure === 'park' || !!t.tourism, 0.8, '공원·관광'],
  ];
  function poiWeight(tags) {
    const t = tags || {};
    for (const [test, w] of RULES) if (test(t)) return w;
    return 0.5;
  }

  // ---------- 시간대 계수 ----------
  function timeFactor(date = new Date()) {
    const h = date.getHours() + date.getMinutes() / 60;
    const wd = date.getDay();
    const weekend = wd === 0 || wd === 6;
    if (!weekend) {
      if (h >= 7.5 && h < 9.5) return { factor: 1.6, label: '출근 시간' };
      if (h >= 11.5 && h < 13.5) return { factor: 1.3, label: '점심 시간' };
      if (h >= 15 && h < 16.5) return { factor: 1.2, label: '하교 시간' };
      if (h >= 17.5 && h < 19.5) return { factor: 1.6, label: '퇴근 시간' };
      if (h >= 22 || h < 6) return { factor: 0.5, label: '심야' };
      return { factor: 1.0, label: '평시' };
    }
    if (h >= 13 && h < 19) return { factor: 1.4, label: '주말 오후' };
    if (h >= 22 || h < 7) return { factor: 0.5, label: '심야' };
    return { factor: 1.0, label: '주말' };
  }

  // 경로(좌표열) 근처(radius m) 지점의 가중합. 한 지점은 한 번만 센다.
  function weightedNear(coords, pois, radius) {
    let sum = 0;
    const kinds = {};
    for (const p of pois) {
      for (let i = 0; i < coords.length; i++) {
        if (Geo.distance(coords[i], p) <= radius) {
          const w = poiWeight(p.tags);
          sum += w;
          const k = kindOf(p.tags);
          kinds[k] = (kinds[k] || 0) + 1;
          break;
        }
      }
    }
    return { sum, kinds };
  }
  function kindOf(tags) {
    const t = tags || {};
    for (const [test, , name] of RULES) if (test(t)) return name;
    return '기타';
  }

  // ---------- 2) 서울 실시간 혼잡도 ----------
  const LEVEL = { '여유': 0, '보통': 1, '약간 붐빔': 2, '붐빔': 3 };
  const LEVEL_PENALTY = [0, 0, 60, 150]; // 초 (경로가 그 장소 반경 안을 지날 때)
  let areas = null; // [{code,name,lat,lon}]
  const cache = new Map(); // name -> {level, pop, ts}

  async function loadAreas() {
    if (areas) return areas;
    try {
      const r = await fetch('seoul_areas.json', { cache: 'no-store' });
      areas = (await r.json()).filter((a) => a.lat && a.lon);
    } catch (_) {
      areas = [];
    }
    return areas;
  }

  // 경로 bbox 근처(여유 300m) 장소 목록
  function areasNear(coords, radius = 300) {
    if (!areas) return [];
    const out = [];
    for (const a of areas) {
      let best = Infinity;
      for (let i = 0; i < coords.length; i += 2) best = Math.min(best, Geo.distance(coords[i], a));
      if (best <= radius) out.push({ ...a, dist: best });
    }
    return out;
  }

  async function fetchSeoulLevel(area, opts) {
    const hit = cache.get(area.name);
    if (hit && Date.now() - hit.ts < 5 * 60 * 1000) return hit;
    let data;
    if (MOCK) {
      const r = await fetch('samples/mock/seoul_ppltn.json', { cache: 'no-store' });
      const all = await r.json();
      data = all[area.name] || null;
      if (!data) return null;
    } else {
      if (!opts.proxy) throw new Error('중계 주소가 없습니다');
      const url = `${opts.proxy}${opts.proxy.includes('?') ? '&' : '?'}area=${encodeURIComponent(area.name)}${opts.key ? '&key=' + encodeURIComponent(opts.key) : ''}`;
      const ctl = new AbortController();
      const t = setTimeout(() => ctl.abort(), 8000);
      const r = await fetch(url, { signal: ctl.signal }).finally(() => clearTimeout(t));
      if (!r.ok) throw new Error('HTTP ' + r.status);
      data = await r.json();
    }
    // citydata_ppltn 응답: SeoulRtd.citydata_ppltn[0].AREA_CONGEST_LVL ...  (중계 서버가 단순화해 줄 수도 있음)
    const row = data.AREA_CONGEST_LVL ? data : (data['SeoulRtd.citydata_ppltn'] || data.SeoulRtd && data.SeoulRtd.citydata_ppltn || [])[0];
    if (!row) return null;
    if (row.AREA_NM && row.AREA_NM !== area.name) return null; // 샘플 키 등 다른 장소 응답은 버림
    const rec = { name: area.name, level: row.AREA_CONGEST_LVL, lv: LEVEL[row.AREA_CONGEST_LVL] ?? 1, pop: [row.AREA_PPLTN_MIN, row.AREA_PPLTN_MAX], msg: row.AREA_CONGEST_MSG, time: row.PPLTN_TIME, source: MOCK ? 'mock' : row.source || 'api', ts: Date.now() };
    cache.set(area.name, rec);
    return rec;
  }

  // 경로 근처 서울 장소들의 혼잡도 조회 (실패는 건너뜀)
  async function seoulLevelsForRoutes(routes, opts) {
    await loadAreas();
    const coords = routes.flatMap((r) => r.coords);
    const near = areasNear(coords, 300);
    const out = [];
    for (const a of near.slice(0, 8)) {
      try {
        const rec = await fetchSeoulLevel(a, opts);
        if (rec) out.push({ ...rec, lat: a.lat, lon: a.lon });
      } catch (e) {
        console.warn('서울 혼잡도 실패', a.name, e.message);
      }
    }
    return out;
  }

  // ---------- 경로 점수 ----------
  // strength: 'weak' | 'mid' | 'strong' → 인파 회피 강도
  const STRENGTH = { weak: 0.5, mid: 1.0, strong: 2.0 };
  function scoreRoute(route, pois, seoulLevels, strength = 'mid', date = new Date()) {
    const km = Math.max(0.05, route.distance / 1000);
    const { sum, kinds } = weightedNear(route.coords, pois, 35);
    const tf = timeFactor(date);
    const index = (sum / km) * tf.factor; // 가중 지점 수 / km × 시간대
    const mult = STRENGTH[strength] || 1;
    let penalty = index * 0.8 * mult; // 지수 1 ≈ 0.8초 가산 (중 기준)
    const seoul = [];
    for (const s of seoulLevels || []) {
      let best = Infinity;
      for (let i = 0; i < route.coords.length; i += 2) best = Math.min(best, Geo.distance(route.coords[i], s));
      if (best <= 300) {
        seoul.push({ name: s.name, level: s.level, dist: Math.round(best) });
        penalty += LEVEL_PENALTY[s.lv] * mult;
      }
    }
    return { index: Math.round(index), raw: Math.round(sum), kinds, time: tf, penalty: Math.round(penalty), seoul };
  }

  return { MOCK, poiWeight, timeFactor, scoreRoute, loadAreas, areasNear, seoulLevelsForRoutes, fetchSeoulLevel, LEVEL, STRENGTH, state: { get areas() { return areas; }, cache } };
})();
