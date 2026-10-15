/*
  route.js — 도보 경로 탐색·점수화·안내 모듈
  - 주소 검색: Nominatim (OpenStreetMap)
  - 도보 경로: OSRM foot 프로파일 (routing.openstreetmap.de, 대안 경로 최대 3개)
  - 경로 점수: 소요 시간 + 횡단보도·계단·기둥 가산 + 인파 지표(POI 밀도) 가산 → 가장 낮은 점수 선택
  - 안내: 경로 위 투영으로 남은 거리/이탈 여부 계산, 다음 회전 지점 음성 안내, 이탈 시 재탐색
*/
const Route = (() => {
  const OSRM = 'https://routing.openstreetmap.de/routed-foot/route/v1/driving/';
  const NOMINATIM = 'https://nominatim.openstreetmap.org/search';

  // 가산 점수(초). 횡단보도 1곳 = 25초 더 걸리는 것과 같게 취급
  const PENALTY = { crosswalk: 25, stairs: 40, pole: 2, poi: 0.8 };
  const OFF_ROUTE_M = 30;   // 경로에서 이만큼 벗어나면 이탈 후보
  const OFF_ROUTE_N = 3;    // 연속 n회 이탈이면 재탐색
  const ANNOUNCE_M = 25;    // 회전 지점 n m 앞에서 안내
  const PASS_M = 10;        // 회전 지점 n m 안이면 지난 것으로

  const state = {
    routes: [], chosen: null, dest: null, from: null,
    active: false, stepIdx: 0, announced: new Set(), offCount: 0,
    lastProgress: null, listeners: [], areaHazards: [], areaPois: [], replanning: false,
  };

  function emit(type, payload) {
    for (const f of state.listeners) {
      try { f(type, payload, state); } catch (e) { console.error(e); }
    }
  }
  function on(f) { state.listeners.push(f); }

  // ?mock=1 : 네트워크 대신 samples/mock/ 의 저장된 응답 사용 (오프라인 시연·테스트)
  const MOCK = new URLSearchParams(location.search).get('mock') === '1';

  // ---------------- 주소 검색 ----------------
  async function geocode(q) {
    if (MOCK) {
      return [
        { name: `${q} (테스트: 서울역, 한강대로, 용산구)`, lat: 37.5529, lon: 126.9726, type: 'station' },
        { name: '서울시청 (테스트)', lat: 37.5665, lon: 126.978, type: 'townhall' },
      ];
    }
    const u = `${NOMINATIM}?format=json&limit=5&countrycodes=kr&accept-language=ko&q=${encodeURIComponent(q)}`;
    const r = await fetch(u, { headers: { Accept: 'application/json' } });
    if (!r.ok) throw new Error('주소 검색 실패 (HTTP ' + r.status + ')');
    const js = await r.json();
    return js.map((x) => ({ name: x.display_name, lat: parseFloat(x.lat), lon: parseFloat(x.lon), type: x.type }));
  }

  // ---------------- 경로 요청 ----------------
  async function fetchRoutes(from, to) {
    const u = MOCK
      ? 'samples/mock/osrm_route.json'
      : `${OSRM}${from.lon},${from.lat};${to.lon},${to.lat}?alternatives=3&steps=true&overview=full&geometries=geojson`;
    const r = await fetch(u, MOCK ? { cache: 'no-store' } : undefined);
    if (!r.ok) throw new Error('경로 서버 오류 (HTTP ' + r.status + ')');
    const js = await r.json();
    if (js.code !== 'Ok' || !js.routes || !js.routes.length) throw new Error('도보 경로를 찾지 못했습니다');
    return js.routes.map((rt) => ({
      distance: rt.distance,
      duration: rt.duration,
      coords: rt.geometry.coordinates.map((c) => ({ lat: c[1], lon: c[0] })),
      steps: rt.legs.flatMap((l) => l.steps).map((s) => ({
        type: s.maneuver.type, modifier: s.maneuver.modifier || '', name: s.name || '',
        lat: s.maneuver.location[1], lon: s.maneuver.location[0], distance: s.distance,
      })),
    }));
  }

  function bboxOf(routes, pad = 0.0015) {
    let s = 90, n = -90, w = 180, e = -180;
    for (const r of routes) for (const c of r.coords) {
      s = Math.min(s, c.lat); n = Math.max(n, c.lat); w = Math.min(w, c.lon); e = Math.max(e, c.lon);
    }
    return { s: s - pad, n: n + pad, w: w - pad, e: e + pad };
  }

  // 경로 꼭짓점에서 radius m 안에 있는 지점 수
  function nearCount(coords, points, radius) {
    let n = 0;
    for (const p of points) {
      for (let i = 0; i < coords.length; i++) {
        if (Geo.distance(coords[i], p) <= radius) { n++; break; }
      }
    }
    return n;
  }

  function score(route, hazards, pois) {
    const cw = nearCount(route.coords, hazards.filter((h) => h.type === 'crosswalk'), 15);
    const st = nearCount(route.coords, hazards.filter((h) => h.type === 'stairs'), 12);
    const po = nearCount(route.coords, hazards.filter((h) => h.type === 'pole'), 6);
    const crowd = nearCount(route.coords, pois, 35);
    const km = route.distance / 1000;
    const penalty = cw * PENALTY.crosswalk + st * PENALTY.stairs + po * PENALTY.pole + crowd * PENALTY.poi;
    return { total: route.duration + penalty, cw, st, po, crowd, crowdPerKm: km > 0 ? Math.round(crowd / km) : 0, penalty: Math.round(penalty) };
  }

  // from → to 경로 계획. opts.considerHazards=false 면 최단 시간만
  async function plan(from, to, opts = {}) {
    const routes = await fetchRoutes(from, to);
    let hazards = [], pois = [];
    if (opts.considerHazards !== false) {
      const bb = bboxOf(routes);
      const span = Geo.distance({ lat: bb.s, lon: bb.w }, { lat: bb.n, lon: bb.e });
      if (span < 6000) { // 너무 넓은 영역은 데이터 요청을 생략
        try { hazards = await Geo.fetchHazardsBbox(bb); } catch (e) { console.warn('위험물 데이터 실패', e); }
        try { pois = await Geo.fetchPois(bb); } catch (e) { console.warn('POI 데이터 실패', e); }
      }
    }
    routes.forEach((r) => {
      r.score = score(r, hazards, pois);
      // 각 안내 지점이 경로 좌표열의 몇 번째 점인지 기록 (통과 판정용)
      for (const s of r.steps) {
        let bi = 0, bd = Infinity;
        for (let i = 0; i < r.coords.length; i++) {
          const d = Geo.distance(s, r.coords[i]);
          if (d < bd) { bd = d; bi = i; }
        }
        s.coordIdx = bi;
      }
    });
    routes.sort((a, b) => a.score.total - b.score.total);
    Object.assign(state, { routes, chosen: routes[0], dest: to, from, active: true, stepIdx: 0, offCount: 0, areaHazards: hazards, areaPois: pois, lastProgress: null });
    state.announced = new Set();
    emit('planned', routes[0]);
    return routes;
  }

  function stopNav() {
    state.active = false;
    state.chosen = null;
    emit('stopped');
  }

  // ---------------- 진행 상황 ----------------
  // 위경도를 기준점 a 주변의 평면(m)으로 바꿔 선분에 투영
  function projectOnSegment(p, a, b) {
    const kx = 111320 * Math.cos((a.lat * Math.PI) / 180);
    const ky = 110540;
    const px = (p.lon - a.lon) * kx, py = (p.lat - a.lat) * ky;
    const bx = (b.lon - a.lon) * kx, by = (b.lat - a.lat) * ky;
    const len2 = bx * bx + by * by;
    let t = len2 > 0 ? (px * bx + py * by) / len2 : 0;
    t = Math.max(0, Math.min(1, t));
    const qx = bx * t, qy = by * t;
    const dist = Math.hypot(px - qx, py - qy);
    return { dist, t, point: { lat: a.lat + (qy / ky), lon: a.lon + (qx / kx) } };
  }

  const DIR_KO = { left: '왼쪽', 'slight left': '왼쪽', 'sharp left': '왼쪽', right: '오른쪽', 'slight right': '오른쪽', 'sharp right': '오른쪽', straight: '직진', uturn: '유턴' };

  function instruction(step, dist) {
    const d = Math.round(dist / 5) * 5;
    const near = d <= 12 ? '지금' : `${d}m 앞에서`;
    const name = step.name ? ` (${step.name})` : '';
    switch (step.type) {
      case 'depart': return `출발합니다${name}`;
      case 'arrive': return '목적지에 도착했습니다';
      case 'turn': case 'end of road': case 'fork': case 'merge': case 'on ramp': case 'off ramp': case 'new name': case 'continue': {
        const ko = DIR_KO[step.modifier] || '직진';
        if (ko === '직진') return `${near} 계속 직진하세요${name}`;
        return `${near} ${ko}으로 가세요${name}`;
      }
      case 'roundabout': case 'rotary': return `${near} 로터리를 지나세요`;
      default: return `${near} ${DIR_KO[step.modifier] || '직진'}${name}`;
    }
  }

  // 현재 위치 갱신 → 진행 정보. 음성 안내/이탈 감지 이벤트 발생
  function update(pos) {
    const r = state.chosen;
    if (!state.active || !r || !pos) return null;
    let best = { dist: Infinity, idx: 0, t: 0, point: r.coords[0] };
    for (let i = 0; i < r.coords.length - 1; i++) {
      const pr = projectOnSegment(pos, r.coords[i], r.coords[i + 1]);
      if (pr.dist < best.dist) best = { dist: pr.dist, idx: i, t: pr.t, point: pr.point };
    }
    let remaining = Geo.distance(best.point, r.coords[best.idx + 1]);
    for (let i = best.idx + 1; i < r.coords.length - 1; i++) remaining += Geo.distance(r.coords[i], r.coords[i + 1]);
    const toDest = Geo.distance(pos, state.dest);

    // 도착
    if (toDest < 15 || (remaining < 15 && best.dist < OFF_ROUTE_M)) {
      const p = { remaining: 0, offRoute: best.dist, next: null, arrived: true };
      state.lastProgress = p;
      state.active = false;
      emit('arrived', p);
      emit('say', '목적지에 도착했습니다');
      return p;
    }

    // 이탈 감지
    if (best.dist > OFF_ROUTE_M) state.offCount += 1; else state.offCount = 0;
    if (state.offCount >= OFF_ROUTE_N && !state.replanning) {
      state.offCount = 0;
      state.replanning = true;
      emit('say', '경로를 벗어났습니다. 다시 찾습니다');
      plan(pos, state.dest, { considerHazards: true })
        .then(() => emit('rerouted', state.chosen))
        .catch((e) => emit('error', e))
        .finally(() => { state.replanning = false; });
    }

    // 다음 안내 지점: 경로 위 진행 위치(best.idx)를 지난 지점은 통과한 것으로 본다
    const steps = r.steps;
    const passed = (s) => s.coordIdx < best.idx || (s.coordIdx === best.idx && best.t > 0.5) || Geo.distance(pos, s) < PASS_M;
    while (state.stepIdx < steps.length - 1 && passed(steps[state.stepIdx])) state.stepIdx += 1;
    let next = steps[state.stepIdx];
    let dNext = next ? Geo.distance(pos, next) : Infinity;
    if (next && state.stepIdx > 0 && dNext < ANNOUNCE_M && !state.announced.has(state.stepIdx)) {
      state.announced.add(state.stepIdx);
      emit('say', instruction(next, dNext));
    }
    const p = { remaining, offRoute: best.dist, next: next ? { text: instruction(next, dNext), dist: dNext, step: next } : null, arrived: false, idx: best.idx };
    state.lastProgress = p;
    emit('progress', p);
    return p;
  }

  // 테스트: 경로를 따라 자동으로 걷는 시뮬레이션 (초당 stepM 미터)
  let simTimer = null;
  function simulateWalk(stepM = 8, intervalMs = 1000) {
    const r = state.chosen;
    if (!r) return;
    clearInterval(simTimer);
    let i = 0, t = 0;
    simTimer = setInterval(() => {
      if (!state.active || i >= r.coords.length - 1) { clearInterval(simTimer); return; }
      const a = r.coords[i], b = r.coords[i + 1];
      const segLen = Geo.distance(a, b) || 1;
      t += stepM / segLen;
      while (t >= 1 && i < r.coords.length - 2) { t -= 1; i += 1; }
      const cur = r.coords[i], nxt = r.coords[Math.min(i + 1, r.coords.length - 1)];
      const lat = cur.lat + (nxt.lat - cur.lat) * Math.min(t, 1);
      const lon = cur.lon + (nxt.lon - cur.lon) * Math.min(t, 1);
      Geo.simulate(lat, lon, Geo.bearing(cur, nxt));
    }, intervalMs);
  }
  function stopSimulation() { clearInterval(simTimer); }

  // ---------------- 음성 ----------------
  function speak(text) {
    if (!('speechSynthesis' in window)) return false;
    try {
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.lang = 'ko-KR';
      u.rate = 1.0;
      window.speechSynthesis.speak(u);
      return true;
    } catch (_) { return false; }
  }

  return { state, on, geocode, plan, stopNav, update, instruction, simulateWalk, stopSimulation, speak, PENALTY };
})();
