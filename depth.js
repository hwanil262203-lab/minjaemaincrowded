/*
  depth.js — 거리 추정으로 '전방 막힘'(벽·닫힌 문·막다른 길) 감지
  - Depth Anything V2 Metric (Indoor, small) ONNX 를 약 1초에 1회 실행해 픽셀마다 거리(미터)를 얻는다.
    상대 거리 모델은 "가까운 것 대비 비율"만 나와 벽을 구분할 수 없었음(발 앞 바닥에 눌려 벽이 '멀게' 나옴).
    미터 단위 모델이라야 "가운데가 2m 안"을 말할 수 있다. 실외에서는 절대값이 조금 틀리지만 가깝다/멀다 구분은 된다.
  - 판정: 화면 가운데 영역(가로 30~70%, 세로 25~75%)의 거리 중앙값 < NEAR_M 이고
    위쪽 영역(세로 0~25%)도 < TOP_M (벽처럼 위까지 막혀 있음) 이면 막힘.
    가운데 20% 지점 거리 < CLOSE_M 이면(코앞에 무언가) 위쪽과 상관없이 막힘.
  - 모델 입력: pixel_values [1,3,266,266] 고정 (ImageNet 정규화), 출력: predicted_depth [1,266,266] (m)
  - 보정(2026-10-16, 샘플 사진, 266 모델): 닫힌 문 2.2m/위 2.1m → 막힘, 복도 2.6m/위 4.6m, 계단 4~6m, 길 16m → 열림
*/
// ONNX Runtime Web 은 한 번에 한 세션만 실행할 수 있다 (동시에 부르면 'Session mismatch').
// 감지 모델과 거리 모델의 run 을 모두 이 줄을 통해 차례로 실행한다.
const OrtQueue = {
  p: Promise.resolve(),
  run(fn) {
    const r = this.p.then(fn, fn);
    this.p = r.catch(() => {});
    return r;
  },
};

const Depth = (() => {
  const MODELS = { webgpu: 'models/depth_metric_indoor_small_266_fp16.onnx', wasm: 'models/depth_metric_indoor_small_266_q8.onnx' };
  const S = 266; // 모델 입력 한 변 (고정, tools/export_depth.py 로 내보냄). 518 보다 7배 빠르고 거리는 약 10~20% 멀게 나옴
  const NEAR_M = 2.4; // 가운데 중앙값이 이보다 가까우면 막힘 후보 (266 모델 기준; 518 모델이면 2.0)
  const TOP_M = 3.0; // 위쪽도 이보다 가까워야(벽) 막힘
  const CLOSE_M = 1.2; // 가운데 20% 지점이 이보다 가까우면 바로 막힘
  const HEAT_MAX_M = 6; // 거리 지도 그릴 때 색 범위 (0~6m)
  const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];

  const state = {
    ready: false, loading: false, busy: false, provider: null, error: null,
    blocked: false, ts: 0, lastMs: 0, runs: 0,
    center: 0, centerP20: 0, top: 0, floor: 0, map: null, // 거리(m). map: Float32Array (0~1, 가까울수록 1)
    listeners: [],
  };
  let session = null, inputName = 'pixel_values', outputName = 'predicted_depth';
  const canvas = document.createElement('canvas');
  canvas.width = S; canvas.height = S;
  const cctx = canvas.getContext('2d', { willReadFrequently: true });
  const input = new Float32Array(3 * S * S);

  function emit() { for (const f of state.listeners) { try { f(state); } catch (e) { console.error(e); } } }
  function onChange(f) { state.listeners.push(f); }

  async function load(opts = {}) {
    if (state.ready || state.loading) return state.ready;
    state.loading = true; state.error = null;
    const tryOrder = opts.providers || (navigator.gpu && opts.useWebGpu !== false ? ['webgpu', 'wasm'] : ['wasm']);
    for (const ep of tryOrder) {
      try {
        const url = MODELS[ep] || MODELS.wasm;
        const t0 = performance.now();
        session = await ort.InferenceSession.create(url, { executionProviders: [ep], graphOptimizationLevel: 'all' });
        inputName = session.inputNames[0]; outputName = session.outputNames[0];
        // 워밍업
        await OrtQueue.run(() => session.run({ [inputName]: new ort.Tensor('float32', input, [1, 3, S, S]) }));
        state.provider = ep; state.ready = true; state.loading = false;
        state.loadMs = Math.round(performance.now() - t0);
        emit();
        return true;
      } catch (e) {
        console.warn('depth load 실패', ep, e);
        state.error = e;
        session = null;
      }
    }
    state.loading = false;
    emit();
    return false;
  }

  // src: {el, w, h} (video 또는 img). 가운데 정사각형을 잘라 S×S 로 (가로 화면이면 좌우가 잘림)
  function preprocess(src) {
    const side = Math.min(src.w, src.h);
    const sx = (src.w - side) / 2, sy = (src.h - side) / 2;
    cctx.drawImage(src.el, sx, sy, side, side, 0, 0, S, S);
    const { data } = cctx.getImageData(0, 0, S, S);
    const n = S * S;
    for (let i = 0; i < n; i++) {
      input[i] = (data[i * 4] / 255 - MEAN[0]) / STD[0];
      input[n + i] = (data[i * 4 + 1] / 255 - MEAN[1]) / STD[1];
      input[2 * n + i] = (data[i * 4 + 2] / 255 - MEAN[2]) / STD[2];
    }
    return new ort.Tensor('float32', input, [1, 3, S, S]);
  }

  function percentile(arr, p) {
    const a = Float32Array.from(arr).sort();
    return a[Math.min(a.length - 1, Math.floor(p * (a.length - 1)))];
  }

  // 거리 지도(m) → 막힘 판정
  function analyze(d, H, W) {
    const sub = (x0, x1, y0, y1) => {
      const out = [];
      for (let y = Math.floor(y0 * H); y < Math.floor(y1 * H); y += 3)
        for (let x = Math.floor(x0 * W); x < Math.floor(x1 * W); x += 3) out.push(d[y * W + x]);
      return out;
    };
    const c = sub(0.3, 0.7, 0.25, 0.75);
    const center = percentile(c, 0.5), centerP20 = percentile(c, 0.2);
    const top = percentile(sub(0.3, 0.7, 0.0, 0.25), 0.5);
    const floor = percentile(sub(0.2, 0.8, 0.9, 1.0), 0.5);
    const blocked = centerP20 < CLOSE_M || (center < NEAR_M && top < TOP_M);
    const norm = new Float32Array(H * W);
    for (let i = 0; i < norm.length; i++) norm[i] = Math.max(0, Math.min(1, 1 - d[i] / HEAT_MAX_M));
    return { blocked, center, centerP20, top, floor, norm };
  }

  async function run(src) {
    if (!state.ready || state.busy || !src || !src.w) return null;
    state.busy = true;
    const t0 = performance.now();
    try {
      const tensor = preprocess(src);
      const out = await OrtQueue.run(() => session.run({ [inputName]: tensor }));
      const o = out[outputName];
      const [H, W] = o.dims.length === 3 ? [o.dims[1], o.dims[2]] : [o.dims[2], o.dims[3]];
      const d = o.data instanceof Float32Array ? o.data : Float32Array.from(o.data);
      const r = analyze(d, H, W);
      state.center = r.center; state.centerP20 = r.centerP20; state.top = r.top; state.floor = r.floor;
      state.blocked = r.blocked;
      state.map = r.norm; state.mapH = H; state.mapW = W;
      state.ts = Date.now(); state.lastMs = Math.round(performance.now() - t0); state.runs += 1;
      emit();
      return r;
    } catch (e) {
      state.error = e;
      console.warn('depth run 실패', e);
      return null;
    } finally {
      state.busy = false;
    }
  }

  // 디버그용: 거리 지도를 작은 캔버스에 색으로 그림 (가까움=빨강, 멂=파랑)
  function draw(cv) {
    if (!state.map || !cv) return;
    const H = state.mapH, W = state.mapW;
    if (cv.width !== W) { cv.width = W; cv.height = H; }
    const ctx = cv.getContext('2d');
    const img = ctx.createImageData(W, H);
    for (let i = 0; i < W * H; i++) {
      const v = state.map[i];
      img.data[i * 4] = Math.round(255 * v);
      img.data[i * 4 + 1] = Math.round(80 * (1 - Math.abs(v - 0.5) * 2));
      img.data[i * 4 + 2] = Math.round(255 * (1 - v));
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    // 판정 영역 표시
    ctx.strokeStyle = state.blocked ? '#ff3b3b' : '#ffffff';
    ctx.lineWidth = 2;
    ctx.strokeRect(0.3 * W, 0.25 * H, 0.4 * W, 0.5 * H);
  }

  // 최근 결과가 유효한가 (maxAgeMs 안에 판정된 것)
  // 느린 기기에서는 한 번 판정이 몇 초 걸리므로 다음 결과가 올 때까지(실행 시간의 2.5배) 유효로 본다
  function recent(maxAgeMs) { const win = Math.max(maxAgeMs || 2500, state.lastMs * 2.5); return state.ts && Date.now() - state.ts < win; }

  return { state, load, run, draw, recent, onChange, NEAR_M, TOP_M, CLOSE_M, S };
})();
