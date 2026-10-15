/*
  geo.js — 위치(GPS)·진행 방향·지도 위험물(OpenStreetMap) 보조 모듈
  - navigator.geolocation 으로 현재 위치/속도/진행 방향을 추적
  - 나침반(DeviceOrientation)으로 정지 상태의 방향 보완
  - Overpass API 로 주변의 횡단보도·계단·기둥(가로등/전봇대) 위치를 받아 캐시
  - 진행 방향 앞에 어떤 위험물이 몇 m 앞에 있는지 계산 (hazardsAhead)
  - 테스트: ?lat=37.56&lon=126.97&heading=45 로 가짜 위치 사용, Geo.simulate() 로 이동
*/
const Geo = (() => {
  const state = {
    pos: null,          // { lat, lon, accuracy, speed, ts }
    heading: null,      // 진행 방향 (도, 북=0, 시계방향)
    headingSrc: null,   // 'gps' | 'compass' | 'fake'
    watchId: null,
    fake: false,
    hazards: [],        // 주변 위험물 [{type, lat, lon, tags}]
    hazardCenter: null, // 마지막으로 위험물을 받아온 중심 좌표
    hazardTs: 0,
    fetching: false,
    lastErr: null,
    listeners: [],
  };

  const R = 6371000;
  const toRad = (d) => (d * Math.PI) / 180;
  const toDeg = (r) => (r * 180) / Math.PI;

  function distance(a, b) {
    const dLat = toRad(b.lat - a.lat);
    const dLon = toRad(b.lon - a.lon);
    const s = Math.sin(dLat / 2) ** 2 + Math.cos(toRad(a.lat)) * Math.cos(toRad(b.lat)) * Math.sin(dLon / 2) ** 2;
    return 2 * R * Math.asin(Math.sqrt(s));
  }

  function bearing(a, b) {
    const y = Math.sin(toRad(b.lon - a.lon)) * Math.cos(toRad(b.lat));
    const x = Math.cos(toRad(a.lat)) * Math.sin(toRad(b.lat)) - Math.sin(toRad(a.lat)) * Math.cos(toRad(b.lat)) * Math.cos(toRad(b.lon - a.lon));
    return (toDeg(Math.atan2(y, x)) + 360) % 360;
  }

  // a - b 를 -180~180 으로
  function angleDiff(a, b) {
    return ((a - b + 540) % 360) - 180;
  }

  function emit() {
    for (const f of state.listeners) {
      try { f(state); } catch (e) { console.error(e); }
    }
  }

  function onChange(f) {
    state.listeners.push(f);
  }

  function setPos(lat, lon, accuracy, speed, gpsHeading) {
    state.pos = { lat, lon, accuracy: accuracy || 0, speed: speed || 0, ts: Date.now() };
    // 움직일 때는 GPS 진행 방향이 가장 정확하다
    if (gpsHeading != null && !Number.isNaN(gpsHeading) && (speed || 0) > 0.4) {
      state.heading = gpsHeading;
      state.headingSrc = 'gps';
    }
    maybeFetchHazards();
    emit();
  }

  function start() {
    const q = new URLSearchParams(location.search);
    if (q.get('lat') && q.get('lon')) {
      state.fake = true;
      state.heading = parseFloat(q.get('heading') || '0');
      state.headingSrc = 'fake';
      setPos(parseFloat(q.get('lat')), parseFloat(q.get('lon')), 5, 1.2, null);
      return Promise.resolve('fake');
    }
    if (!('geolocation' in navigator)) return Promise.reject(new Error('이 기기는 위치를 지원하지 않습니다'));
    return new Promise((resolve, reject) => {
      let settled = false;
      state.watchId = navigator.geolocation.watchPosition(
        (p) => {
          setPos(p.coords.latitude, p.coords.longitude, p.coords.accuracy, p.coords.speed, p.coords.heading);
          if (!settled) { settled = true; resolve('gps'); }
        },
        (e) => {
          state.lastErr = e;
          if (!settled) { settled = true; reject(new Error(e.message || '위치 권한이 거부되었습니다')); }
        },
        { enableHighAccuracy: true, maximumAge: 2000, timeout: 15000 }
      );
    });
  }

  function stop() {
    if (state.watchId != null && 'geolocation' in navigator) navigator.geolocation.clearWatch(state.watchId);
    state.watchId = null;
  }

  // 나침반: 반드시 사용자 탭(클릭) 안에서 호출해야 iOS 권한 창이 뜬다
  async function enableCompass() {
    try {
      if (typeof DeviceOrientationEvent !== 'undefined' && typeof DeviceOrientationEvent.requestPermission === 'function') {
        const r = await DeviceOrientationEvent.requestPermission();
        if (r !== 'granted') return false;
      }
    } catch (_) {
      return false;
    }
    const handler = (e) => {
      let h = null;
      if (typeof e.webkitCompassHeading === 'number') h = e.webkitCompassHeading; // iOS
      else if (e.absolute && typeof e.alpha === 'number') h = (360 - e.alpha) % 360; // Android
      if (h == null || state.fake) return;
      // 걷는 중(GPS 방향 유효)이 아닐 때만 나침반 사용
      const moving = state.pos && (state.pos.speed || 0) > 0.4 && state.headingSrc === 'gps' && Date.now() - state.pos.ts < 4000;
      if (!moving) {
        state.heading = h;
        state.headingSrc = 'compass';
      }
    };
    window.addEventListener('deviceorientationabsolute', handler, true);
    window.addEventListener('deviceorientation', handler, true);
    return true;
  }

  // 테스트용: 위치를 직접 지정 (방향은 이전 위치에서 계산)
  function simulate(lat, lon, heading) {
    state.fake = true;
    if (heading != null) {
      state.heading = heading;
    } else if (state.pos) {
      const d = distance(state.pos, { lat, lon });
      if (d > 1) state.heading = bearing(state.pos, { lat, lon });
    }
    state.headingSrc = 'fake';
    setPos(lat, lon, 5, 1.3, null);
  }

  // ---------------- Overpass (OpenStreetMap 데이터) ----------------
  const MIRRORS = [
    'https://overpass-api.de/api/interpreter',
    'https://overpass.kumi.systems/api/interpreter',
    'https://overpass.private.coffee/api/interpreter',
  ];

  // ?mock=1 : 네트워크 대신 samples/mock/ 의 저장된 응답 사용 (오프라인 시연·테스트)
  const MOCK = new URLSearchParams(location.search).get('mock') === '1';

  async function overpass(query) {
    if (MOCK) {
      const file = query.includes('"shop"') ? 'samples/mock/overpass_pois.json' : 'samples/mock/overpass_hazards.json';
      const res = await fetch(file, { cache: 'no-store' });
      if (!res.ok) throw new Error('mock 파일 없음: ' + file);
      await new Promise((r) => setTimeout(r, 300)); // 네트워크 지연 흉내
      return await res.json();
    }
    let lastErr = null;
    for (const url of MIRRORS) {
      try {
        const ctl = new AbortController();
        const t = setTimeout(() => ctl.abort(), 25000);
        const res = await fetch(url, {
          method: 'POST',
          body: 'data=' + encodeURIComponent(query),
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
          signal: ctl.signal,
        });
        clearTimeout(t);
        if (!res.ok) throw new Error('HTTP ' + res.status);
        return await res.json();
      } catch (e) {
        lastErr = e;
      }
    }
    throw lastErr || new Error('Overpass 실패');
  }

  const HAZARD_SEL = (area) =>
    `node["highway"="crossing"]${area};node["footway"="crossing"]${area};way["footway"="crossing"]${area};` +
    `way["highway"="steps"]${area};node["highway"="street_lamp"]${area};node["man_made"="utility_pole"]${area};node["power"="pole"]${area};`;

  function classify(el) {
    const t = el.tags || {};
    if (t.highway === 'crossing' || t.footway === 'crossing') return 'crosswalk';
    if (t.highway === 'steps') return 'stairs';
    if (t.highway === 'street_lamp' || t.man_made === 'utility_pole' || t.power === 'pole') return 'pole';
    return null;
  }

  function parseHazards(data) {
    const out = [];
    for (const el of data.elements || []) {
      const type = classify(el);
      if (!type) continue;
      const lat = el.lat != null ? el.lat : el.center && el.center.lat;
      const lon = el.lon != null ? el.lon : el.center && el.center.lon;
      if (lat == null) continue;
      out.push({ type, lat, lon, id: el.id, tags: el.tags || {} });
    }
    return out;
  }

  // 현재 위치 주변(반경 m)의 위험물 → state.hazards
  async function fetchHazards(lat, lon, radius = 200) {
    const q = `[out:json][timeout:20];(${HAZARD_SEL(`(around:${radius},${lat},${lon})`)});out center;`;
    const data = await overpass(q);
    state.hazards = parseHazards(data);
    state.hazardCenter = { lat, lon };
    state.hazardTs = Date.now();
    emit();
    return state.hazards;
  }

  // 임의의 사각 영역(경로 전체)의 위험물 → 반환만 (state 는 건드리지 않음)
  async function fetchHazardsBbox(bb) {
    const area = `(${bb.s},${bb.w},${bb.n},${bb.e})`;
    const q = `[out:json][timeout:25];(${HAZARD_SEL(area)});out center 3000;`;
    return parseHazards(await overpass(q));
  }

  // 인파 대용 지표: 상점·음식점·역·정류장 등 사람이 모이는 지점
  async function fetchPois(bb) {
    const area = `(${bb.s},${bb.w},${bb.n},${bb.e})`;
    const q =
      `[out:json][timeout:25];(node["shop"]${area};` +
      `node["amenity"~"^(restaurant|cafe|fast_food|bar|pub|marketplace|bus_station|school|university|cinema|theatre)$"]${area};` +
      `node["public_transport"~"^(station|stop_position|platform)$"]${area};` +
      `node["railway"~"^(station|subway_entrance)$"]${area};);out skel qt 2000;`;
    const data = await overpass(q);
    return (data.elements || []).filter((e) => e.lat != null).map((e) => ({ lat: e.lat, lon: e.lon }));
  }

  function maybeFetchHazards() {
    if (!state.pos || state.fetching) return;
    if (state.hazardCenter && distance(state.pos, state.hazardCenter) < 90 && Date.now() - state.hazardTs < 10 * 60 * 1000) return;
    state.fetching = true;
    fetchHazards(state.pos.lat, state.pos.lon)
      .catch((e) => { state.lastErr = e; })
      .finally(() => { state.fetching = false; });
  }

  // 진행 방향 앞(±halfAngle°)의 위험물. 아주 가까우면(6m) 방향과 무관하게 포함
  function hazardsAhead({ maxDist = 35, halfAngle = 45 } = {}) {
    if (!state.pos || state.heading == null) return [];
    const res = [];
    for (const h of state.hazards) {
      const d = distance(state.pos, h);
      if (d > maxDist) continue;
      const rel = angleDiff(bearing(state.pos, h), state.heading);
      if (Math.abs(rel) > halfAngle && d > 6) continue;
      res.push({ ...h, dist: d, rel, dir: rel < -20 ? 'left' : rel > 20 ? 'right' : 'center' });
    }
    res.sort((a, b) => a.dist - b.dist);
    return res;
  }

  return { state, start, stop, enableCompass, simulate, distance, bearing, angleDiff, fetchHazards, fetchHazardsBbox, fetchPois, hazardsAhead, onChange };
})();
