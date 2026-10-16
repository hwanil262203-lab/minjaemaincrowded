/*
  YOLO 객체 감지 (ONNX Runtime Web) + 스마트폰 진동
  - best.onnx (Ultralytics YOLO export, opset 12) 를 브라우저에서 실행
  - WebGPU 우선, 실패 시 WASM(CPU) 폴백
  - 박스 위치로 방향(좌/중/우), 박스 높이로 근접도 계산
  - 신뢰도 임계치 + 연속 프레임 히스테리시스 후 방향/근접도별 진동 패턴
*/

const CONFIG = {
  // ?model=yolo11n.onnx 처럼 주소로 모델 파일을 바꿔 테스트할 수 있다
  modelUrl: new URLSearchParams(location.search).get('model') || './best.onnx',
  labelsUrl: new URLSearchParams(location.search).get('labels') || './labels.json',
  // best.onnx 가 없을 때 자동으로 쓰는 시험용 모델 (YOLOv10n, COCO 80클래스 사전학습)
  fallbackModelUrl: './yolov10n.onnx',
  fallbackLabelsUrl: './labels_coco.json',
  ortVersion: '1.30.0',
  nmsIou: 0.45,
  maxDraw: 8,
  vibrateRepeatMs: 1500, // 위험이 계속되면 이 간격으로 진동을 다시 울림
  // GPS 보조: 지도상 위험물이 진행 방향 앞에 있으면
  gpsLowConf: 0.1,                 // 그 클래스는 이 낮은 신뢰도까지 감지로 인정
  gpsExpectDist: 35,               // 이 거리(m) 안이면 '예상됨'
  gpsAlertDist: { crosswalk: 15, stairs: 10, pole: 6 }, // 이 거리 안이면 모델이 못 봐도 경고
  // 감지 루프 자가 복구
  frameTimeoutMs: 8000, // 한 프레임 추론이 이보다 오래 걸리면 멈춘 것으로 보고 세션 재생성
  loadTimeoutMs: 60000, // 모델 로드·워밍업 제한 시간
  maxConsecutiveErrors: 3, // 연속 오류가 이만큼이면 세션 재생성
  maxRecoveries: 3, // 빠른 세션 재생성 한도 (넘으면 모델 전체 재로드)
  maxTotalRecoveries: 30, // 전체 재로드 한도 (넘으면 오류 창)
  modelLoadAttempts: 30, // 모델 로드 자동 재시도 횟수
  modelLoadRetryMs: 3000, // 재시도 간격
};

// 클래스별 한국어 이름, 위험 가중치(0~1), 색. 키는 labels.json 의 영어 이름.
const CLASS_INFO = {
  crosswalk: { ko: '횡단보도', weight: 0.8, color: '#facc15' },
  stairs: { ko: '계단', weight: 1.0, color: '#f97316' },
  pole: { ko: '기둥·가로등', weight: 1.0, color: '#ef4444' },
  person: { ko: '사람', weight: 0.6, color: '#38bdf8' },
  vehicle: { ko: '차량', weight: 0.9, color: '#a855f7' },
  blocked: { ko: '전방 막힘', weight: 1.0, color: '#ff3b3b' }, // 거리 추정(depth.js)으로 만든 가상 항목: 벽·닫힌 문·막다른 길
  // COCO 사전학습 모델로 파이프라인을 테스트할 때 쓰이는 보조 항목
  car: { ko: '자동차', weight: 0.9, color: '#a855f7' },
  bus: { ko: '버스', weight: 0.9, color: '#a855f7' },
  truck: { ko: '트럭', weight: 0.9, color: '#a855f7' },
  bicycle: { ko: '자전거', weight: 0.7, color: '#a855f7' },
  motorcycle: { ko: '오토바이', weight: 0.8, color: '#a855f7' },
  'fire hydrant': { ko: '소화전', weight: 0.7, color: '#ef4444' },
  bench: { ko: '벤치', weight: 0.6, color: '#ef4444' },
  'traffic light': { ko: '신호등', weight: 0.2, color: '#94a3b8' },
  'stop sign': { ko: '정지표지', weight: 0.2, color: '#94a3b8' },
  dog: { ko: '개', weight: 0.5, color: '#38bdf8' },
  chair: { ko: '의자', weight: 0.5, color: '#ef4444' },
};
const DEFAULT_CLASS = { weight: 0.5, color: '#94a3b8' };
const DIR_KO = { left: '왼쪽', center: '정면', right: '오른쪽' };

const ui = {
  video: document.getElementById('video'),
  overlay: document.getElementById('overlay'),
  btnCamera: document.getElementById('btnCamera'),
  btnLoadModel: document.getElementById('btnLoadModel'),
  btnStart: document.getElementById('btnStart'),
  btnStop: document.getElementById('btnStop'),
  threshold: document.getElementById('threshold'),
  steadyFrames: document.getElementById('steadyFrames'),
  proximity: document.getElementById('proximity'),
  status: document.getElementById('status'),
  engine: document.getElementById('engine'),
  log: document.getElementById('log'),
  detections: document.getElementById('detections'),
  floatingIndicator: document.getElementById('floatingIndicator'),
};

let session = null;
let labels = { names: [], imgsz: 640, nms_in_model: null };
let inputName = null;
let outputName = null;
let inputSize = 640;
let engineName = '-';
let usingFallbackModel = false;
let webcamStream = null;
let testImage = null; // ?image=... 로 지정한 정지 이미지 (카메라 없이 테스트)
let isRunning = false;
let animationHandle = null;

const log = (msg) => {
  const t = new Date().toLocaleTimeString();
  ui.log.textContent += `[${t}] ${msg}\n`;
  ui.log.scrollTop = ui.log.scrollHeight;
};

function setStatus(text) {
  ui.status.textContent = text;
}

function classInfo(label) {
  return CLASS_INFO[label] || { ...DEFAULT_CLASS, ko: label };
}

// ============================================
// 카메라
// ============================================
async function openCamera() {
  if (webcamStream) {
    log('카메라가 이미 실행 중입니다.');
    return;
  }
  try {
    log('카메라 권한 요청 중...');
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    ui.video.srcObject = webcamStream;
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('비디오 로드 타임아웃')), 10000);
      const ready = () => {
        clearTimeout(timeout);
        ui.overlay.width = ui.video.videoWidth;
        ui.overlay.height = ui.video.videoHeight;
        resolve();
      };
      ui.video.onloadedmetadata = ready;
      if (ui.video.readyState >= 1) ready();
    });
    await ui.video.play();
    log(`✓ 카메라 시작됨 (${ui.video.videoWidth}x${ui.video.videoHeight})`);
  } catch (e) {
    log(`✗ 카메라 오류: ${e.message}`);
    console.error('카메라 오류:', e);
  }
}

// 테스트 모드: ?image=sample.jpg 로 정지 이미지를 카메라 대신 사용
async function loadTestImage(url) {
  try {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = () => reject(new Error('이미지를 불러오지 못했습니다'));
      img.src = url;
    });
    testImage = img;
    ui.video.style.display = 'none';
    img.style.cssText = 'width:100%;height:auto;display:block;max-width:640px';
    ui.video.parentElement.insertBefore(img, ui.overlay);
    ui.overlay.width = img.naturalWidth;
    ui.overlay.height = img.naturalHeight;
    log(`✓ 테스트 이미지 사용 (${img.naturalWidth}x${img.naturalHeight}): ${url}`);
  } catch (e) {
    log(`✗ 테스트 이미지 오류: ${e.message}`);
  }
}

// ============================================
// 모델 로드 (ONNX Runtime Web)
// ============================================
async function loadModel(opts = {}) {
  if (modelLoading) {
    log('모델을 이미 불러오는 중입니다.');
    return false;
  }
  if (session && !opts.force) {
    log('모델이 이미 준비되어 있습니다.');
    return true;
  }
  modelLoading = true;
  setModelButton('loading');
  try {
    setStatus('모델 로딩 중...');
    ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${CONFIG.ortVersion}/dist/`;
    ort.env.wasm.numThreads = Math.max(1, Math.min(4, navigator.hardwareConcurrency || 1));
    if (session) {
      const old = session;
      session = null;
      try { await old.release(); } catch (_) { /* 무시 */ }
    }

    // labels.json (없어도 동작: 클래스 이름이 숫자로 표시됨)
    try {
      const r = await fetch(CONFIG.labelsUrl, { cache: 'no-store' });
      if (r.ok) labels = { ...labels, ...(await r.json()) };
    } catch (_) {
      /* 무시 */
    }
    // 모델 파일이 없으면(아직 학습 전) 시험용 사전학습 모델로 대체
    let modelUrl = CONFIG.modelUrl;
    usingFallbackModel = false;
    try {
      const head = await fetch(modelUrl, { method: 'HEAD', cache: 'no-store' });
      if (!head.ok) throw new Error(`HTTP ${head.status}`);
    } catch (e) {
      log(`⚠ ${modelUrl} 없음 (${e.message}) → 시험용 모델 ${CONFIG.fallbackModelUrl} 사용`);
      modelUrl = CONFIG.fallbackModelUrl;
      usingFallbackModel = true;
      try {
        const r = await fetch(CONFIG.fallbackLabelsUrl, { cache: 'no-store' });
        if (r.ok) labels = { ...labels, ...(await r.json()) };
      } catch (_) {
        /* 무시 */
      }
    }
    activeModelUrl = modelUrl;
    log(`모델 파일: ${modelUrl}`);

    // WebGPU 우선, 실패하면 WASM. (opts.wasmOnly 면 처음부터 WASM)
    const providers = opts.providers || (navigator.gpu && settings.useWebGpu && !opts.wasmOnly ? ['webgpu', 'wasm'] : ['wasm']);
    if (navigator.gpu && !settings.useWebGpu) log('이 기기에서는 WebGPU 를 건너뛰고 WASM 으로 실행합니다 (설정에서 바꿀 수 있음)');
    await createSession(modelUrl, providers);

    ui.engine.textContent = engineLabel();
    log(`클래스: ${labels.names.join(', ') || '(labels.json 없음)'}`);
    setStatus(usingFallbackModel ? '시험용 모델 준비 완료' : '모델 준비 완료');
    setModelButton('ready');
    loopState.recoveries = 0;
    return true;
  } catch (e) {
    log(`✗ 모델 로드 실패: ${e.message}`);
    console.error('모델 로드 오류:', e);
    setStatus('모델 로드 실패 — 🔄 모델 로드 버튼을 눌러 다시 시도');
    session = null;
    setModelButton('retry');
    return false;
  } finally {
    modelLoading = false;
  }
}

let modelLoading = false;
let activeModelUrl = null;

function withTimeout(promise, ms, label) {
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error(`${label} 시간 초과 (${ms}ms)`)), ms);
    promise.then(
      (v) => { clearTimeout(t); resolve(v); },
      (e) => { clearTimeout(t); reject(e); }
    );
  });
}

function engineLabel() {
  return `${engineName.toUpperCase()} · ${inputSize}px · NMS ${labels.nms_in_model ? '내장' : 'JS'}` +
    (usingFallbackModel ? ' · 시험용 모델' : '');
}

function setModelButton(state) {
  const b = ui.btnLoadModel;
  if (!b) return;
  if (state === 'loading') {
    b.disabled = true;
    b.textContent = '⏳ 모델 로드 중...';
  } else if (state === 'ready') {
    b.disabled = false;
    b.textContent = '🔄 모델 다시 로드';
    b.classList.add('secondary');
  } else {
    b.disabled = false;
    b.textContent = '🔄 모델 로드';
    b.classList.remove('secondary'); // 실패 시 눈에 띄게
  }
}

// 세션 생성 + 워밍업. providers 를 순서대로 시도해 처음 성공한 것을 쓴다
async function createSession(modelUrl, providers) {
  let lastErr = null;
  for (const ep of providers) {
    try {
      const s = await withTimeout(
        ort.InferenceSession.create(modelUrl, { executionProviders: [ep], graphOptimizationLevel: 'all' }),
        CONFIG.loadTimeoutMs, `${ep} 세션 생성`
      );
      const inName = s.inputNames[0];
      const outName = s.outputNames[0];
      // 입력 크기: 모델 메타데이터 → labels.json → 640 순서로 결정
      let size = labels.imgsz || 640;
      const meta = s.inputMetadata && s.inputMetadata[0];
      const dims = meta && (meta.shape || meta.dims);
      if (dims && typeof dims[2] === 'number' && dims[2] > 0) size = dims[2];
      // 워밍업 겸 출력 형식 확인
      const zeros = new ort.Tensor('float32', new Float32Array(3 * size * size), [1, 3, size, size]);
      const t0 = performance.now();
      const out = await withTimeout(OrtQueue.run(() => s.run({ [inName]: zeros })), CONFIG.loadTimeoutMs, `${ep} 워밍업`);
      const od = out[outName].dims;
      session = s;
      engineName = ep;
      inputName = inName;
      outputName = outName;
      inputSize = size;
      labels.nms_in_model = od.length === 3 && od[2] === 6;
      log(`✓ 모델 로드 완료 (${ep}, 입력 ${size}px, 출력 [${od.join(',')}], 워밍업 ${Math.round(performance.now() - t0)}ms)`);
      return true;
    } catch (e) {
      lastErr = e;
      log(`${ep} 초기화 실패 → 다음 백엔드 시도 (${String((e && e.message) || e).split('\n')[0]})`);
    }
  }
  throw lastErr || new Error('세션 생성 실패');
}

// 모델이 로드될 때까지 반복 시도. 한도를 넘으면 오류 창을 띄운다.
let modelRetryPromise = null;

function showErrorModal(text) {
  const m = document.getElementById('errorModal');
  if (!m) return;
  document.getElementById('errorModalText').textContent = text;
  m.hidden = false;
}

function ensureModel(maxAttempts = CONFIG.modelLoadAttempts, delayMs = CONFIG.modelLoadRetryMs) {
  if (session) return Promise.resolve(true);
  if (modelRetryPromise) return modelRetryPromise; // 이미 재시도 중이면 그 결과를 함께 기다림
  modelRetryPromise = (async () => {
    try {
      for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        if (session) return true;
        if (attempt > 1) {
          setStatus(`모델 로드 재시도 ${attempt}/${maxAttempts}...`);
          log(`모델 로드 재시도 ${attempt}/${maxAttempts}`);
        }
        const ok = await loadModel();
        if (ok && session) return true;
        await new Promise((r) => setTimeout(r, delayMs));
      }
      log(`✗ 모델 로드 ${maxAttempts}회 실패`);
      setStatus('모델 로드 실패');
      showErrorModal('모델 로드에 실패했습니다. 다시 접속하여 주십시오.');
      return false;
    } finally {
      modelRetryPromise = null;
    }
  })();
  return modelRetryPromise;
}

// 추론이 멈추거나 오류가 반복되면 세션을 새로 만든다 (WebGPU 가 문제면 WASM 으로 내려감)
async function recoverSession(reason) {
  loopState.recoveries += 1;
  if (loopState.recoveries > CONFIG.maxRecoveries) {
    log(`✗ 복구 한도(${CONFIG.maxRecoveries}회) 초과`);
    return false;
  }
  log(`⚠ ${reason} → 모델을 WASM 으로 다시 불러옵니다 (복구 ${loopState.recoveries}/${CONFIG.maxRecoveries})`);
  const old = session;
  session = null;
  try { if (old) await old.release(); } catch (_) { /* 무시 */ }
  try {
    await createSession(activeModelUrl || CONFIG.modelUrl, ['wasm']);
    ui.engine.textContent = engineLabel() + ' · 복구됨';
    return true;
  } catch (e) {
    log(`✗ 복구 실패: ${e.message}`);
    return false;
  }
}

// ============================================
// 전처리: letterbox → NCHW float32 (0~1)
// ============================================
const pre = { canvas: document.createElement('canvas'), ctx: null };

// 현재 프레임 소스 (카메라 또는 테스트 이미지)
function frameSource() {
  if (testImage) return { el: testImage, w: testImage.naturalWidth, h: testImage.naturalHeight };
  return { el: ui.video, w: ui.video.videoWidth, h: ui.video.videoHeight };
}

function preprocess(src, S) {
  if (pre.canvas.width !== S) {
    pre.canvas.width = S;
    pre.canvas.height = S;
    pre.ctx = pre.canvas.getContext('2d', { willReadFrequently: true });
  }
  const vw = src.w;
  const vh = src.h;
  const scale = Math.min(S / vw, S / vh);
  const nw = Math.round(vw * scale);
  const nh = Math.round(vh * scale);
  const dx = Math.floor((S - nw) / 2);
  const dy = Math.floor((S - nh) / 2);
  const ctx = pre.ctx;
  ctx.fillStyle = '#727272';
  ctx.fillRect(0, 0, S, S);
  ctx.drawImage(src.el, 0, 0, vw, vh, dx, dy, nw, nh);
  const { data } = ctx.getImageData(0, 0, S, S);
  const area = S * S;
  const f = new Float32Array(3 * area);
  for (let i = 0, p = 0; i < area; i++, p += 4) {
    f[i] = data[p] / 255;
    f[area + i] = data[p + 1] / 255;
    f[2 * area + i] = data[p + 2] / 255;
  }
  return { tensor: new ort.Tensor('float32', f, [1, 3, S, S]), scale, dx, dy };
}

// ============================================
// 후처리: 출력 텐서 → [{label, score, x1,y1,x2,y2}] (비디오 좌표)
// ============================================
function iou(a, b) {
  const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const inter = ix * iy;
  const ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return ua > 0 ? inter / ua : 0;
}

function nms(cands, thr) {
  cands.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const c of cands) {
    if (keep.every((k) => k.cls !== c.cls || iou(k, c) < thr)) keep.push(c);
  }
  return keep;
}

function postprocess(output, lb, conf, vw, vh) {
  const d = output.dims;
  const data = output.data;
  let boxes = [];

  if (d.length === 3 && d[2] === 6) {
    // Ultralytics nms=True 출력: [1, max_det, 6] = x1,y1,x2,y2,conf,cls
    for (let i = 0; i < d[1]; i++) {
      const o = i * 6;
      const s = data[o + 4];
      if (s < conf) continue;
      boxes.push({ x1: data[o], y1: data[o + 1], x2: data[o + 2], y2: data[o + 3], score: s, cls: data[o + 5] | 0 });
    }
  } else if (d.length === 3) {
    // 원시 출력: [1, 4+nc, N] (Ultralytics 기본) 또는 [1, N, 4+nc]
    const transposed = d[1] > d[2];
    const C = transposed ? d[2] : d[1];
    const N = transposed ? d[1] : d[2];
    const nc = C - 4;
    const at = transposed ? (c, i) => data[i * C + c] : (c, i) => data[c * N + i];
    const cands = [];
    for (let i = 0; i < N; i++) {
      let best = 0;
      let bc = -1;
      for (let c = 0; c < nc; c++) {
        const s = at(4 + c, i);
        if (s > best) {
          best = s;
          bc = c;
        }
      }
      if (best < conf) continue;
      const cx = at(0, i);
      const cy = at(1, i);
      const w = at(2, i);
      const h = at(3, i);
      cands.push({ x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2, score: best, cls: bc });
    }
    boxes = nms(cands, CONFIG.nmsIou);
  } else {
    throw new Error(`지원하지 않는 출력 형식: [${d.join(',')}]`);
  }

  // letterbox 좌표 → 비디오 좌표
  const clampX = (v) => Math.max(0, Math.min(vw, v));
  const clampY = (v) => Math.max(0, Math.min(vh, v));
  return boxes.map((b) => ({
    label: labels.names[b.cls] !== undefined ? labels.names[b.cls] : String(b.cls),
    score: b.score,
    x1: clampX((b.x1 - lb.dx) / lb.scale),
    y1: clampY((b.y1 - lb.dy) / lb.scale),
    x2: clampX((b.x2 - lb.dx) / lb.scale),
    y2: clampY((b.y2 - lb.dy) / lb.scale),
  }));
}

// ============================================
// 위험 판단: 방향 + 근접도 + 클래스 가중치
// ============================================
function assess(dets, vw, vh, minProximity) {
  for (const det of dets) {
    const info = classInfo(det.label);
    det.ko = info.ko;
    det.color = info.color;
    det.prox = (det.y2 - det.y1) / vh; // 박스 높이 비율 = 근접도 (0~1)
    const cx = (det.x1 + det.x2) / 2 / vw;
    det.dir = cx < 0.35 ? 'left' : cx > 0.65 ? 'right' : 'center';
    det.near = det.prox >= 0.5;
    det.risk = det.score * info.weight * Math.min(1, det.prox / 0.5);
  }
  dets.sort((a, b) => b.risk - a.risk);
  const best = dets.find((d) => d.prox >= minProximity && classInfo(d.label).weight >= 0.5);
  return best || null;
}

const hysteresisState = { lastLabel: null, steadyCount: 0 };

function updateHysteresis(currentLabel, requiredFrames) {
  if (currentLabel && currentLabel === hysteresisState.lastLabel) {
    hysteresisState.steadyCount += 1;
  } else if (currentLabel) {
    hysteresisState.lastLabel = currentLabel;
    hysteresisState.steadyCount = 1;
  } else {
    hysteresisState.lastLabel = null;
    hysteresisState.steadyCount = 0;
  }
  return hysteresisState.steadyCount >= requiredFrames ? hysteresisState.lastLabel : null;
}

// ============================================
// 사용자 설정 (localStorage 에 저장)
// ============================================
const SETTINGS_KEY = 'hazard-settings-v1';
const settings = {
  vibStrength: 'mid', // weak | mid | strong
  flash: true, // 위험 알림 중 화면 깜빡임
  flashLevel: 'light', // light | mid | strong
  onboarded: false, // 첫 접속 안내를 봤는지
  gpsAssist: true, // 지도 데이터로 위험물 예측
  depthAssist: true, // 거리 추정으로 전방 막힘(벽·막다른 길) 감지, 1초에 1회
  depthWebGpu: true, // 거리 모델의 WebGPU 사용 여부 (감지 모델과 별개. 실패하면 자동으로 false 저장)
  navEnabled: true, // 도보 경로 안내
  voice: true, // 음성 안내
  crowdAvoid: 'mid', // 인파 회피 강도: off | weak | mid | strong
  seoulLive: false, // 서울 실시간 혼잡도(API 키·중계 서버 필요)
  seoulProxy: '', // 비우면 자동: Netlify 는 /.netlify/functions/seoul, 로컬은 http://127.0.0.1:8002/seoul
  seoulKey: '', // 서울열린데이터광장 인증키 (사용자 입력, 이 기기에만 저장)
  useWebGpu: true, // WebGPU 시도 여부. 기기에서 실패하면 자동으로 false 로 저장
};

// 인파 회피 설정 → Route.plan 에 넘길 옵션
function crowdOptions() {
  const local = ['localhost', '127.0.0.1', '[::1]'].includes(location.hostname) || location.protocol === 'file:';
  const proxy = settings.seoulProxy.trim() || (local ? 'http://127.0.0.1:8002/seoul' : '/.netlify/functions/seoul');
  return { strength: settings.crowdAvoid, seoul: { enabled: settings.seoulLive, proxy, key: settings.seoulKey.trim() } };
}

function loadSettings() {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (raw) Object.assign(settings, JSON.parse(raw));
  } catch (_) {
    /* 사생활 모드 등에서는 무시 */
  }
}

function saveSettings() {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  } catch (_) {
    /* 무시 */
  }
}

const VIB_STRENGTH = {
  // 웹 진동 API 는 강도 조절이 없어 펄스 길이·반복 간격으로 흉내 낸다
  weak: { scale: 0.5, repeatMs: 2200 },
  mid: { scale: 1.0, repeatMs: 1500 },
  strong: { scale: 1.6, repeatMs: 1000 },
};
const FLASH_OPACITY = { light: 0.22, mid: 0.38, strong: 0.55 };

// ============================================
// 진동: 방향 = 펄스 개수(좌1/정면2/우3), 근접 = 펄스 길이(가까움 150ms / 보통 350ms)
// ============================================
const vib = { key: null, t: 0 };
const canVibrate = 'vibrate' in navigator;

function vibrationPattern(dir, near, strength = settings.vibStrength) {
  const s = VIB_STRENGTH[strength] || VIB_STRENGTH.mid;
  const pulses = dir === 'left' ? 1 : dir === 'center' ? 2 : 3;
  const len = Math.round((near ? 150 : 350) * s.scale);
  const gap = 120;
  const pattern = [];
  for (let i = 0; i < pulses; i++) {
    pattern.push(len);
    if (i < pulses - 1) pattern.push(gap);
  }
  return pattern;
}

function vibrateFor(det) {
  const key = `${det.label}|${det.dir}|${det.near}`;
  const now = performance.now();
  const repeatMs = (VIB_STRENGTH[settings.vibStrength] || VIB_STRENGTH.mid).repeatMs;
  if (key !== vib.key || now - vib.t > repeatMs) {
    if (canVibrate) navigator.vibrate(vibrationPattern(det.dir, det.near));
    vib.key = key;
    vib.t = now;
  }
  setFlash(true);
}

function stopVibration() {
  if (vib.key && canVibrate) navigator.vibrate(0);
  vib.key = null;
  vib.t = 0;
  setFlash(false);
}

// 화면 깜빡임 (진동 모드/무음 대비)
const flashOverlay = document.getElementById('flashOverlay');

function setFlash(on) {
  if (!flashOverlay) return;
  if (on && settings.flash) {
    flashOverlay.style.setProperty('--flash-opacity', FLASH_OPACITY[settings.flashLevel] || FLASH_OPACITY.light);
    flashOverlay.classList.add('active');
  } else {
    flashOverlay.classList.remove('active');
  }
}

function testVibration() {
  if (!canVibrate) return false;
  // 정면·가까움 패턴 (2회) 으로 체감
  return navigator.vibrate(vibrationPattern('center', true));
}

function testFlash() {
  const prev = settings.flash;
  settings.flash = true;
  setFlash(true);
  setTimeout(() => {
    setFlash(false);
    settings.flash = prev;
  }, 2000);
}

// ============================================
// 화면 표시
// ============================================
function drawDetections(ctx, dets, best) {
  const W = ctx.canvas.width;
  const H = ctx.canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.font = '14px ui-monospace, monospace';
  ctx.textBaseline = 'top';
  for (const det of dets.slice(0, CONFIG.maxDraw)) {
    const isBest = det === best;
    ctx.lineWidth = isBest ? 4 : 2;
    ctx.strokeStyle = det.color;
    ctx.setLineDash(det.source === 'gps' || det.source === 'depth' ? [10, 6] : []);
    ctx.strokeRect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);
    ctx.setLineDash([]);
    const text = det.source === 'depth' ? `${det.ko} (벽·막다른 길) ${det.depthM.toFixed(1)}m` : det.source === 'gps' ? `${det.ko} GPS ${Math.round(det.gpsDist)}m` : `${det.ko} ${(det.score * 100).toFixed(0)}%${det.gpsBoost ? ' +GPS' : ''}`;
    const tw = ctx.measureText(text).width + 8;
    ctx.fillStyle = det.color;
    ctx.fillRect(det.x1, Math.max(0, det.y1 - 18), tw, 18);
    ctx.fillStyle = '#0b1020';
    ctx.fillText(text, det.x1 + 4, Math.max(0, det.y1 - 18) + 2);
  }
  if (best) {
    ctx.font = 'bold 18px system-ui, sans-serif';
    const text = `⚠ ${best.ko} · ${DIR_KO[best.dir]} · ${best.near ? '가까움' : '접근 중'}`;
    const tw = ctx.measureText(text).width + 16;
    ctx.fillStyle = 'rgba(239, 68, 68, 0.85)';
    ctx.fillRect(8, 8, tw, 30);
    ctx.fillStyle = '#fff';
    ctx.fillText(text, 16, 14);
  }
}

function updateDetectionList(dets, best) {
  if (!dets.length) {
    ui.detections.innerHTML = '<li class="muted">감지된 물체 없음</li>';
    return;
  }
  ui.detections.innerHTML = dets
    .slice(0, CONFIG.maxDraw)
    .map((d) => {
      const cls = d === best ? 'danger' : '';
      return `<li class="${cls}"><span class="dot" style="background:${d.color}"></span>` +
        `<b>${d.ko}</b> ${d.source === 'depth' ? `<span class="gps">거리 추정 ${d.depthM.toFixed(1)}m</span>` : d.source === 'gps' ? `<span class="gps">GPS ${Math.round(d.gpsDist)}m</span>` : `${(d.score * 100).toFixed(0)}%${d.gpsBoost ? ' <span class="gps">+GPS</span>' : ''}`} · ${DIR_KO[d.dir]} · 근접 ${(d.prox * 100).toFixed(0)}%</li>`;
    })
    .join('');
}

// 상태 인디케이터: ok(초록 ✓) / warn(노랑 !) / danger(빨강 !)
const INDICATOR = {
  ok: { cls: '', text: '✓', title: '정상 작동 중 (드래그하여 이동 가능)' },
  warn: { cls: 'warn', text: '!', title: '비정상: 감지가 동작하지 않습니다 (드래그하여 이동 가능)' },
  danger: { cls: 'danger', text: '!', title: '위험 감지됨! (드래그하여 이동 가능)' },
};
let indicatorState = null;

function setIndicator(state) {
  if (state === indicatorState) return;
  indicatorState = state;
  const s = INDICATOR[state] || INDICATOR.warn;
  ui.floatingIndicator.classList.remove('warn', 'danger');
  if (s.cls) ui.floatingIndicator.classList.add(s.cls);
  ui.floatingIndicator.textContent = s.text;
  ui.floatingIndicator.title = s.title;
}

// 정상 작동 감시: 감지 루프가 멈추거나 프레임이 오래 끊기면 노랑
const health = { lastFrameTs: 0, danger: false };

function updateFloatingIndicator(isDanger) {
  health.danger = isDanger;
  refreshIndicator();
}

function refreshIndicator() {
  if (health.danger) return setIndicator('danger');
  const alive = isRunning && session && performance.now() - health.lastFrameTs < 4000;
  setIndicator(alive ? 'ok' : 'warn');
}

setInterval(refreshIndicator, 1000);

// ============================================
// 추론 루프 (이전 추론이 끝난 뒤 다음 프레임 처리 → 겹치지 않음)
// ============================================
const perf = { ms: 0, frames: 0, lastReport: 0 };

// 감지 루프 상태 (멈춤·오류 복구용)
const loopState = { token: 0, inFlight: false, frameStart: 0, lastTick: 0, consecutiveErrors: 0, recoveries: 0, totalRecoveries: 0, okStreak: 0, lastVideoTime: -1, lastVideoChange: 0 };

async function inferenceLoop() {
  if (!session) {
    log('✗ 모델이 준비되지 않았습니다. 🔄 모델 로드 버튼을 눌러 주세요.');
    setStatus('모델 없음 — 🔄 모델 로드 필요');
    setModelButton('retry');
    return;
  }
  if (!(webcamStream || testImage)) {
    log('✗ 카메라가 준비되지 않았습니다. 📷 카메라 시작을 눌러 주세요.');
    return;
  }
  if (isRunning) return;
  isRunning = true;
  loopState.token += 1;
  const myToken = loopState.token;
  loopState.consecutiveErrors = 0;
  loopState.lastTick = performance.now();
  setStatus('감지 중...');
  log('감지 시작됨');
  const ctx = ui.overlay.getContext('2d');

  const step = async () => {
    if (!isRunning || myToken !== loopState.token) return;
    const t0 = performance.now();
    loopState.inFlight = true;
    loopState.frameStart = t0;
    loopState.lastTick = t0;
    try {
      const src = frameSource();
      const vw = src.w;
      const vh = src.h;
      if (vw && vh && session) {
        if (ui.overlay.width !== vw) {
          ui.overlay.width = vw;
          ui.overlay.height = vh;
        }
        const conf = parseFloat(ui.threshold.value) || 0.25;
        const requiredFrames = parseInt(ui.steadyFrames.value, 10) || 3;
        const minProximity = parseFloat(ui.proximity.value) || 0.2;

        const lb = preprocess(src, inputSize);
        const result = await withTimeout(OrtQueue.run(() => session.run({ [inputName]: lb.tensor })), CONFIG.frameTimeoutMs, '추론');
        if (!isRunning || myToken !== loopState.token) return;
        const dets = detectWithGps(result[outputName], lb, conf, vw, vh);
        addDepthDetection(dets, vw, vh);
        const best = assess(dets, vw, vh, minProximity);
        const confirmed = updateHysteresis(best ? best.label : null, requiredFrames);

        drawDetections(ctx, dets, best);
        updateDetectionList(dets, best);
        updateFloatingIndicator(!!confirmed);

        if (confirmed && best) vibrateFor(best);
        else stopVibration();
        health.lastFrameTs = performance.now();
        loopState.consecutiveErrors = 0;
        loopState.okStreak += 1;
        if (loopState.okStreak === 100) loopState.recoveries = 0; // 한동안 잘 돌면 복구 횟수 초기화
      }
    } catch (e) {
      loopState.consecutiveErrors += 1;
      loopState.okStreak = 0;
      const msg = String((e && e.message) || e).split('\n')[0];
      console.error('추론 오류:', e);
      log(`감지 오류 ${loopState.consecutiveErrors}회 (${engineName}): ${msg}`);
      const timedOut = /시간 초과/.test(msg);
      const gpuErr = engineName === 'webgpu' && /webgpu|kernel|gpu|unsupported/i.test(msg);
      if (gpuErr && settings.useWebGpu) {
        settings.useWebGpu = false;
        saveSettings();
        log('⚠ 이 기기의 WebGPU 가 모델을 실행하지 못합니다. 앞으로는 처음부터 WASM 으로 실행합니다.');
      }
      if (timedOut || gpuErr || loopState.consecutiveErrors >= CONFIG.maxConsecutiveErrors) {
        const ok = await recoverSession(timedOut ? '추론이 응답하지 않음' : gpuErr ? 'WebGPU 실행 실패' : '오류 반복');
        if (!ok) {
          // 빠른 복구가 막히면 정지하지 않고 모델 전체를 다시 불러온 뒤 자동으로 재시작
          loopState.inFlight = false;
          stopLoop();
          loopState.totalRecoveries += 1;
          if (loopState.totalRecoveries > CONFIG.maxTotalRecoveries) {
            setStatus('감지 중단');
            showErrorModal('감지를 계속할 수 없습니다. 다시 접속하여 주십시오.');
            return;
          }
          setStatus('모델을 다시 불러오는 중...');
          log(`↻ 모델 전체 재로드 후 자동 재시작 (${loopState.totalRecoveries}/${CONFIG.maxTotalRecoveries})`);
          loopState.recoveries = 0;
          const stale = session;
          session = null; // 기존 세션을 버리고 완전히 새로 불러온다
          if (stale) stale.release().catch(() => {});
          ensureModel().then((loaded) => {
            if (loaded && (webcamStream || testImage)) inferenceLoop();
          });
          return;
        }
        loopState.consecutiveErrors = 0;
      }
    } finally {
      loopState.inFlight = false;
    }
    if (!isRunning || myToken !== loopState.token) return;
    // 성능 표시
    perf.ms = perf.ms * 0.8 + (performance.now() - t0) * 0.2;
    perf.frames += 1;
    if (performance.now() - perf.lastReport > 1000) {
      perf.lastReport = performance.now();
      setStatus(`감지 중 · ${Math.round(perf.ms)}ms/프레임 (${(1000 / Math.max(1, perf.ms)).toFixed(1)} FPS)`);
    }
    scheduleNext(step);
  };
  step();
}

// 다음 프레임 예약: requestAnimationFrame 이 오지 않는 환경(일부 모바일 브라우저, 백그라운드)에서는 타이머로 진행
function scheduleNext(step) {
  let fired = false;
  const go = () => {
    if (fired) return;
    fired = true;
    step();
  };
  animationHandle = requestAnimationFrame(go);
  setTimeout(go, document.hidden ? 1000 : 300);
}

function stopLoop() {
  isRunning = false;
  loopState.token += 1;
  stopVibration();
  updateFloatingIndicator(false);
  setStatus('정지됨');
  if (animationHandle) cancelAnimationFrame(animationHandle);
}

function reopenCamera(why) {
  log(`⚠ ${why} → 카메라를 다시 엽니다`);
  try { if (webcamStream) webcamStream.getTracks().forEach((t) => t.stop()); } catch (_) { /* 무시 */ }
  webcamStream = null;
  ui.video.srcObject = null;
  openCamera();
}

// 감시: 루프가 멈추거나 카메라 영상이 얼면 스스로 복구
setInterval(() => {
  if (!isRunning) return;
  const now = performance.now();
  // 1) 추론 Promise 가 끝내 돌아오지 않음 (withTimeout 도 못 잡는 극단 상황)
  if (loopState.inFlight && now - loopState.frameStart > CONFIG.frameTimeoutMs + 15000) {
    log('⚠ 감지 루프가 응답하지 않아 다시 시작합니다');
    isRunning = false;
    loopState.token += 1;
    loopState.inFlight = false;
    inferenceLoop();
    return;
  }
  // 2) 다음 프레임이 예약되지 않은 채 멈춤
  if (!loopState.inFlight && now - loopState.lastTick > 6000 && !document.hidden) {
    log('⚠ 감지가 멈춘 것을 감지해 다시 시작합니다');
    isRunning = false;
    loopState.token += 1;
    inferenceLoop();
    return;
  }
  // 3) 카메라 스트림이 끊기거나 영상이 얼어 있음
  if (webcamStream && !testImage) {
    const v = ui.video;
    const track = webcamStream.getVideoTracks()[0];
    if (track && track.readyState === 'ended') {
      reopenCamera('카메라 스트림이 끊김');
      return;
    }
    if (v.paused && v.readyState >= 2) v.play().catch(() => {});
    if (v.currentTime !== loopState.lastVideoTime) {
      loopState.lastVideoTime = v.currentTime;
      loopState.lastVideoChange = now;
    } else if (loopState.lastVideoChange && now - loopState.lastVideoChange > 5000 && !document.hidden) {
      loopState.lastVideoChange = now;
      reopenCamera('카메라 영상이 5초 이상 멈춤');
    }
  }
}, 2000);

ui.btnCamera.addEventListener('click', openCamera);
ui.btnLoadModel.addEventListener('click', () => (session ? loadModel({ force: true }) : ensureModel()));
document.getElementById('btnErrorReload').addEventListener('click', () => location.reload());
ui.btnStart.addEventListener('click', inferenceLoop);
ui.btnStop.addEventListener('click', stopLoop);

// ============================================
// 드래그 가능한 Floating Indicator
// ============================================
let isDragging = false;
let dragStartX = 0;
let dragStartY = 0;
let indicatorStartX = 0;
let indicatorStartY = 0;

function beginDrag(clientX, clientY) {
  isDragging = true;
  dragStartX = clientX;
  dragStartY = clientY;
  const rect = ui.floatingIndicator.getBoundingClientRect();
  indicatorStartX = rect.left;
  indicatorStartY = rect.top;
  ui.floatingIndicator.classList.add('dragging');
}

function moveDrag(clientX, clientY) {
  if (!isDragging) return;
  ui.floatingIndicator.style.left = indicatorStartX + (clientX - dragStartX) + 'px';
  ui.floatingIndicator.style.top = indicatorStartY + (clientY - dragStartY) + 'px';
  ui.floatingIndicator.style.right = 'auto';
}

function endDrag() {
  if (!isDragging) return;
  isDragging = false;
  ui.floatingIndicator.classList.remove('dragging');
}

ui.floatingIndicator.addEventListener('mousedown', (e) => {
  beginDrag(e.clientX, e.clientY);
  e.preventDefault();
});
document.addEventListener('mousemove', (e) => moveDrag(e.clientX, e.clientY));
document.addEventListener('mouseup', endDrag);
ui.floatingIndicator.addEventListener('touchstart', (e) => {
  beginDrag(e.touches[0].clientX, e.touches[0].clientY);
  e.preventDefault();
});
document.addEventListener('touchmove', (e) => {
  if (!isDragging) return;
  moveDrag(e.touches[0].clientX, e.touches[0].clientY);
  e.preventDefault();
}, { passive: false });
document.addEventListener('touchend', endDrag);

// ============================================
// 시작 안내 창 + 설정 창
// ============================================
const panels = {
  start: document.getElementById('startOverlay'),
  settings: document.getElementById('settingsPanel'),
};

function syncSettingsUI() {
  document.querySelectorAll('input[name="vibStrength"]').forEach((r) => (r.checked = r.value === settings.vibStrength));
  document.querySelectorAll('input[name="flashLevel"]').forEach((r) => (r.checked = r.value === settings.flashLevel));
  document.getElementById('flashToggle').checked = settings.flash;
  document.getElementById('startFlashToggle').checked = settings.flash;
  document.getElementById('gpsAssistToggle').checked = settings.gpsAssist;
  document.getElementById('depthToggle').checked = settings.depthAssist;
  document.getElementById('navToggle').checked = settings.navEnabled;
  document.getElementById('voiceToggle').checked = settings.voice;
  document.querySelectorAll('input[name="crowdAvoid"]').forEach((r) => (r.checked = r.value === settings.crowdAvoid));
  document.getElementById('seoulLiveToggle').checked = settings.seoulLive;
  document.getElementById('seoulProxy').value = settings.seoulProxy;
  document.getElementById('seoulKey').value = settings.seoulKey;
  document.getElementById('seoulFields').hidden = !settings.seoulLive;
  document.getElementById('webgpuToggle').checked = settings.useWebGpu;
}

function bindSettingsUI() {
  document.querySelectorAll('input[name="vibStrength"]').forEach((r) =>
    r.addEventListener('change', () => {
      if (r.checked) {
        settings.vibStrength = r.value;
        saveSettings();
      }
    })
  );
  document.querySelectorAll('input[name="flashLevel"]').forEach((r) =>
    r.addEventListener('change', () => {
      if (r.checked) {
        settings.flashLevel = r.value;
        saveSettings();
      }
    })
  );
  document.getElementById('flashToggle').addEventListener('change', (e) => {
    settings.flash = e.target.checked;
    saveSettings();
    if (!settings.flash) setFlash(false);
  });
  document.getElementById('depthToggle').addEventListener('change', (e) => {
    settings.depthAssist = e.target.checked;
    saveSettings();
    if (settings.depthAssist) ensureDepth();
    else renderDepthStatus();
  });
  document.getElementById('gpsAssistToggle').addEventListener('change', (e) => {
    settings.gpsAssist = e.target.checked;
    saveSettings();
    if (settings.gpsAssist && !Geo.state.pos) startGeo();
  });
  document.getElementById('navToggle').addEventListener('change', (e) => {
    settings.navEnabled = e.target.checked;
    saveSettings();
    document.getElementById('nav').hidden = !settings.navEnabled;
    if (!settings.navEnabled) Route.stopNav();
    if (settings.navEnabled && !Geo.state.pos) startGeo();
  });
  document.getElementById('voiceToggle').addEventListener('change', (e) => {
    settings.voice = e.target.checked;
    saveSettings();
  });
  document.querySelectorAll('input[name="crowdAvoid"]').forEach((r) =>
    r.addEventListener('change', () => {
      if (r.checked) {
        settings.crowdAvoid = r.value;
        saveSettings();
      }
    })
  );
  document.getElementById('seoulLiveToggle').addEventListener('change', (e) => {
    settings.seoulLive = e.target.checked;
    document.getElementById('seoulFields').hidden = !settings.seoulLive;
    saveSettings();
  });
  document.getElementById('seoulProxy').addEventListener('change', (e) => {
    settings.seoulProxy = e.target.value.trim();
    saveSettings();
  });
  document.getElementById('seoulKey').addEventListener('change', (e) => {
    settings.seoulKey = e.target.value.trim();
    saveSettings();
  });
  document.getElementById('btnSeoulTest').addEventListener('click', async () => {
    const out = document.getElementById('seoulTestResult');
    out.textContent = '확인 중...';
    try {
      await Crowd.loadAreas();
      const opts = crowdOptions().seoul;
      const area = Crowd.state.areas.find((a) => a.name === '광화문·덕수궁') || Crowd.state.areas[0];
      Crowd.state.cache.delete(area.name);
      const rec = await Crowd.fetchSeoulLevel(area, opts);
      out.textContent = rec
        ? `✓ ${rec.name}: ${rec.level} (${rec.pop[0]}~${rec.pop[1]}명, ${rec.time || '-'}) · ${Crowd.MOCK ? '저장된 테스트 응답(mock)' : '중계: ' + opts.proxy}`
        : '✗ 응답에 혼잡도가 없습니다';
    } catch (e) {
      out.textContent = `✗ 실패: ${e.message} · 중계 서버가 켜져 있는지, 인증키가 맞는지 확인하세요`;
    }
  });
  document.getElementById('webgpuToggle').addEventListener('change', (e) => {
    settings.useWebGpu = e.target.checked;
    saveSettings();
    log(`GPU 가속 ${settings.useWebGpu ? '사용' : '사용 안 함'} — 🔄 모델 다시 로드 를 누르면 적용됩니다`);
  });
  document.getElementById('btnCopyLog').addEventListener('click', async () => {
    const text = [navigator.userAgent, ui.engine.textContent, ui.status.textContent, '', ui.log.textContent].join('\n');
    try {
      await navigator.clipboard.writeText(text);
      log('📋 로그를 복사했습니다. 메시지로 붙여 넣어 보내 주세요.');
    } catch (_) {
      window.prompt('아래 내용을 길게 눌러 복사하세요', text);
    }
  });
  document.getElementById('btnVibTest').addEventListener('click', () => {
    if (!testVibration()) log('진동을 지원하지 않거나 차단된 기기입니다.');
  });
  document.getElementById('btnFlashTest').addEventListener('click', testFlash);
  document.getElementById('btnSettings').addEventListener('click', () => {
    syncSettingsUI();
    panels.settings.hidden = false;
  });
  document.getElementById('btnSettingsClose').addEventListener('click', () => {
    panels.settings.hidden = true;
  });

  // 시작 안내 창
  if (!canVibrate) document.getElementById('noVibrateNote').hidden = false;
  document.getElementById('btnStartApp').addEventListener('click', startApp);
}

function showStartOverlay() {
  panels.start.classList.toggle('first', !settings.onboarded);
  panels.start.classList.toggle('compact', settings.onboarded);
  syncSettingsUI();
  panels.start.hidden = false;
}

// 사용자의 탭(시작하기) 이후에만 카메라·감지를 시작한다.
// Chrome 은 사용자가 페이지를 한 번도 누르지 않으면 navigator.vibrate 를 막는다.
async function startApp() {
  settings.flash = document.getElementById('startFlashToggle').checked;
  settings.onboarded = true;
  saveSettings();
  panels.start.hidden = true;

  // 위치·나침반 (사용자 탭 안에서 요청해야 권한 창이 뜬다)
  document.getElementById('nav').hidden = !settings.navEnabled;
  if (settings.gpsAssist || settings.navEnabled) {
    Geo.enableCompass();
    startGeo();
  }
  if (settings.depthAssist) ensureDepth();

  const imageUrl = new URLSearchParams(location.search).get('image');
  if (imageUrl) {
    if (!testImage) await loadTestImage(imageUrl);
  } else if (!webcamStream) {
    await openCamera();
  }
  // 모델이 아직 없으면(로드 실패·지연) 한 번 더 시도
  if (!session) await ensureModel();
  if (session && (webcamStream || testImage)) {
    log('🚀 감지를 시작합니다...');
    inferenceLoop();
  } else if (!session) {
    log('✗ 모델이 없어 시작하지 못했습니다. 🔄 모델 로드 버튼을 눌러 주세요.');
    setModelButton('retry');
  } else {
    log('✗ 카메라가 준비되지 않았습니다. 📷 카메라 시작을 눌러 주세요.');
  }
}

// ============================================
// 초기화: 설정 로드 → 모델 로드 → 시작 안내 창
// ============================================
window.addEventListener('load', async () => {
  loadSettings();
  bindSettingsUI();
  log('앱 시작됨');
  if (typeof ort === 'undefined') {
    log('✗ ONNX Runtime Web 로드 실패 (인터넷 연결 확인)');
    setStatus('ORT 로드 실패');
    return;
  }
  log(`✓ ONNX Runtime Web ${ort.env.versions ? ort.env.versions.web || '' : ''} 로드 완료`);
  showStartOverlay();
  await ensureModel();
});


// ============================================
// GPS 보조 감지: 지도상 위험물이 앞에 있으면 낮은 신뢰도도 인정 + 모델이 못 봐도 경고
// ============================================
const GPS_KO = { crosswalk: '횡단보도', stairs: '계단', pole: '기둥·가로등' };

// ============================================
// 거리 추정(depth.js): 1초에 1회 실행해 전방 막힘(벽·막다른 길) 판단
// ============================================
const depthUI = { status: document.getElementById('depthStatus'), canvas: document.getElementById('depthCanvas') };
let depthTimer = null;

async function ensureDepth() {
  if (Depth.state.ready || Depth.state.loading) return;
  log('거리 추정 모델 로드 중... (Depth Anything V2 small)');
  renderDepthStatus();
  // 감지 모델(YOLO)은 일부 기기에서 WebGPU 커널(GatherND int64) 오류로 WASM 을 쓰지만, 거리 모델은 그 연산이 없어 WebGPU 가 된다 → 따로 판단
  const ok = await Depth.load({ useWebGpu: settings.depthWebGpu });
  if (ok && Depth.state.provider !== 'webgpu' && settings.depthWebGpu && navigator.gpu) {
    settings.depthWebGpu = false;
    saveSettings();
    log('거리 모델: 이 기기의 WebGPU 에서 실패해 앞으로는 WASM 으로 실행합니다');
  }
  if (ok) log(`✓ 거리 추정 모델 로드 완료 (${Depth.state.provider}, 워밍업 ${Depth.state.loadMs}ms) — 1초에 1회 전방 막힘 판정`);
  else log(`✗ 거리 추정 모델 로드 실패: ${Depth.state.error && Depth.state.error.message}`);
  renderDepthStatus();
  if (ok && !depthTimer) {
    depthTimer = setInterval(() => {
      if (!settings.depthAssist || !isRunning || Depth.state.busy) return;
      if (Depth.state.ts && Date.now() - Depth.state.ts < Math.max(1000, Depth.state.lastMs * 1.2) - 50) return; // 1초 간격, 느린 기기는 그만큼 늦춤
      const src = frameSource();
      if (src.w && src.h) Depth.run(src);
    }, 250);
  }
}

Depth.onChange(() => {
  renderDepthStatus();
  Depth.draw(depthUI.canvas);
});

function renderDepthStatus() {
  if (!depthUI.status) return;
  const st = Depth.state;
  if (!settings.depthAssist) depthUI.status.textContent = '전방 거리: 꺼짐';
  else if (st.loading) depthUI.status.textContent = '전방 거리: 모델 로드 중';
  else if (!st.ready) depthUI.status.textContent = '전방 거리: 사용 불가';
  else if (!st.ts) depthUI.status.textContent = `전방 거리: 대기 (${st.provider})`;
  else depthUI.status.textContent = `전방 ${st.blocked ? '막힘 ⚠' : '열림'} · 가운데 ${st.center.toFixed(1)}m · 위 ${st.top.toFixed(1)}m · ${st.lastMs}ms`;
  depthUI.status.style.borderColor = st.blocked && settings.depthAssist ? '#ff3b3b' : '';
}

// 최근 1초 판정이 '막힘'이면 화면 가운데에 가상 박스를 넣어 기존 위험 처리(확정·진동)에 태운다
function addDepthDetection(dets, vw, vh) {
  if (!settings.depthAssist || !Depth.state.blocked || !Depth.recent(2500)) return dets;
  if (dets.some((d) => d.source === 'depth')) return dets;
  // 가까울수록 높은 점수 (2m → 0.5, 1m → 0.75, 0.5m 이하 → 0.9)
  const score = Math.max(0.5, Math.min(0.9, 1 - Depth.state.center / 4));
  dets.push({ label: 'blocked', score, x1: vw * 0.25, y1: vh * 0.15, x2: vw * 0.75, y2: vh, source: 'depth', depthM: Depth.state.center });
  return dets;
}

function detectWithGps(output, lb, conf, vw, vh) {
  if (!settings.gpsAssist || !Geo.state.pos) return postprocess(output, lb, conf, vw, vh);

  const ahead = Geo.hazardsAhead({ maxDist: CONFIG.gpsExpectDist });
  const expected = new Set(ahead.map((h) => h.type));
  const low = Math.min(conf, CONFIG.gpsLowConf);
  const dets = postprocess(output, lb, low, vw, vh).filter((d) => {
    if (d.score >= conf) return true;
    if (expected.has(d.label)) { d.gpsBoost = true; return true; }
    return false;
  });

  // 모델이 전혀 못 봤지만 지도상 아주 가까운 위험물 → 가상 박스로 경고
  for (const h of ahead) {
    const limit = CONFIG.gpsAlertDist[h.type];
    if (limit == null || h.dist > limit) continue;
    if (dets.some((d) => d.label === h.type)) continue;
    const near = 1 - h.dist / limit; // 0(멀다) ~ 1(코앞)
    const bw = vw / 3;
    const x1 = h.dir === 'left' ? 0 : h.dir === 'right' ? vw - bw : (vw - bw) / 2;
    const bh = vh * (0.3 + 0.5 * near);
    dets.push({ label: h.type, score: 0.5 + 0.4 * near, x1, y1: vh - bh, x2: x1 + bw, y2: vh, source: 'gps', gpsDist: h.dist });
  }
  return dets;
}

// ============================================
// 위치 시작 + 길찾기 UI
// ============================================
const navUI = {
  input: document.getElementById('destInput'),
  btnGeocode: document.getElementById('btnGeocode'),
  results: document.getElementById('geoResults'),
  btnStart: document.getElementById('btnNavStart'),
  btnStop: document.getElementById('btnNavStop'),
  btnSim: document.getElementById('btnSimWalk'),
  status: document.getElementById('navStatus'),
  mapEl: document.getElementById('map'),
};
let geoStarted = false;
let selectedDest = null;
let wakeLock = null;
const mapState = { map: null, me: null, dest: null, line: null, hazardLayer: null, seoulLayer: null, altLines: [] };

function startGeo() {
  if (geoStarted) return;
  geoStarted = true;
  Geo.start()
    .then((src) => {
      log(`✓ 위치 사용 시작 (${src === 'fake' ? '테스트 좌표' : 'GPS'})`);
      ensureMap();
    })
    .catch((e) => {
      geoStarted = false;
      log(`✗ 위치 오류: ${e.message}`);
      navUI.status.textContent = `위치를 사용할 수 없습니다: ${e.message}`;
    });
}

function ensureMap() {
  if (mapState.map || typeof L === 'undefined' || !navUI.mapEl || navUI.mapEl.offsetParent === null && document.getElementById('nav').hidden) return;
  const p = Geo.state.pos || { lat: 37.5665, lon: 126.978 };
  mapState.map = L.map(navUI.mapEl, { zoomControl: true }).setView([p.lat, p.lon], 16);
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '© OpenStreetMap' }).addTo(mapState.map);
  mapState.me = L.circleMarker([p.lat, p.lon], { radius: 8, color: '#3356ff', fillColor: '#3356ff', fillOpacity: 0.9 }).addTo(mapState.map);
  mapState.hazardLayer = L.layerGroup().addTo(mapState.map);
  renderHazards();
}

const HZ_COLOR = { crosswalk: '#facc15', stairs: '#f97316', pole: '#ef4444' };
const SEOUL_COLOR = { '여유': '#22c55e', '보통': '#facc15', '약간 붐빔': '#f97316', '붐빔': '#ef4444' };

// 서울 실시간 혼잡도 장소를 지도에 원으로 표시 (반경 300m = 가산 적용 범위)
function renderSeoulAreas() {
  if (!mapState.map) return;
  if (!mapState.seoulLayer) mapState.seoulLayer = L.layerGroup().addTo(mapState.map);
  mapState.seoulLayer.clearLayers();
  for (const s of Route.state.seoulLevels || []) {
    const c = SEOUL_COLOR[s.level] || '#6b7a99';
    L.circle([s.lat, s.lon], { radius: 300, color: c, fillColor: c, fillOpacity: 0.12, weight: 1.5 })
      .bindTooltip(`${s.name}: ${s.level} (${s.pop[0]}~${s.pop[1]}명)`)
      .addTo(mapState.seoulLayer);
  }
}
function renderHazards() {
  if (!mapState.hazardLayer) return;
  mapState.hazardLayer.clearLayers();
  for (const h of Geo.state.hazards) {
    L.circleMarker([h.lat, h.lon], { radius: 4, color: HZ_COLOR[h.type], fillColor: HZ_COLOR[h.type], fillOpacity: 0.8, weight: 1 })
      .bindTooltip(`${GPS_KO[h.type]}`)
      .addTo(mapState.hazardLayer);
  }
}

function fmtDist(m) {
  return m >= 1000 ? `${(m / 1000).toFixed(1)}km` : `${Math.round(m)}m`;
}

function renderNavStatus() {
  const pos = Geo.state.pos;
  const parts = [];
  if (pos) {
    parts.push(`<span class="pill">📍 ${pos.lat.toFixed(5)}, ${pos.lon.toFixed(5)} (±${Math.round(pos.accuracy)}m)</span>`);
    parts.push(`<span class="pill">방향 ${Geo.state.heading == null ? '-' : Math.round(Geo.state.heading) + '° (' + Geo.state.headingSrc + ')'}</span>`);
    parts.push(`<span class="pill">주변 위험물 ${Geo.state.hazards.length}곳</span>`);
    const ahead = Geo.hazardsAhead({ maxDist: CONFIG.gpsExpectDist });
    if (ahead.length) parts.push(`<span class="pill" style="border-color:#facc15">앞 ${ahead.slice(0, 3).map((h) => `${GPS_KO[h.type]} ${Math.round(h.dist)}m ${DIR_KO[h.dir]}`).join(' · ')}</span>`);
  } else {
    parts.push('위치 확인 중...');
  }
  const r = Route.state.chosen;
  if (r && Route.state.active) {
    const pr = Route.state.lastProgress;
    const sc = r.score || {};
    parts.push(`<div style="margin-top:6px">경로 <b>${fmtDist(r.distance)}</b> · 약 ${Math.round(r.duration / 60)}분 · 횡단보도 ${sc.cw ?? '-'}곳 · 계단 ${sc.st ?? '-'}곳 · 인파 지수 ${sc.crowdPerKm ?? '-'}/km (대안 ${Route.state.routes.length}개 중 선택)</div>`);
    const ci = sc.crowdInfo;
    if (ci) {
      const kinds = Object.entries(ci.kinds).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([k, n]) => `${k} ${n}`).join(' · ');
      parts.push(`<div>인파: ${ci.time.label}(×${ci.time.factor}) · 경로 주변 ${ci.raw ? kinds : '사람 모이는 곳 없음'} · 가산 ${ci.penalty}초</div>`);
      if (ci.seoul.length) parts.push(`<div>서울 실시간: ${ci.seoul.map((s) => `<span class="pill" style="border-color:${SEOUL_COLOR[s.level] || '#6b7a99'}">${s.name} ${s.level}</span>`).join(' ')}</div>`);
      else if (settings.seoulLive) parts.push('<div class="muted">서울 실시간: 경로 300m 안에 측정 장소 없음(또는 조회 실패)</div>');
    }
    const alts = Route.state.routes.slice(1);
    if (alts.length) parts.push(`<div class="muted">대안: ${alts.map((a) => `${fmtDist(a.distance)} 점수 ${Math.round(a.score.total)}${a.score.crowdInfo ? ' (인파 ' + a.score.crowdInfo.index + '/km)' : ''}`).join(' · ')} ← 선택 경로 점수 ${Math.round(sc.total)}</div>`);
    if (pr) {
      parts.push(`<div>남은 거리 <b>${fmtDist(pr.remaining)}</b>${pr.offRoute > 30 ? ' · <span style="color:#f97316">경로 이탈 중</span>' : ''}</div>`);
      if (pr.next) parts.push(`<div>다음: ${pr.next.text}</div>`);
    }
  } else if (Route.state.lastProgress && Route.state.lastProgress.arrived) {
    parts.push('<div style="margin-top:6px"><b>목적지에 도착했습니다.</b></div>');
  }
  navUI.status.innerHTML = parts.join(' ');
}

function renderRoute() {
  if (!mapState.map) return;
  for (const l of mapState.altLines) mapState.map.removeLayer(l);
  mapState.altLines = [];
  if (mapState.line) { mapState.map.removeLayer(mapState.line); mapState.line = null; }
  if (mapState.dest) { mapState.map.removeLayer(mapState.dest); mapState.dest = null; }
  renderSeoulAreas();
  const r = Route.state.chosen;
  if (!r) return;
  for (const alt of Route.state.routes.slice(1)) {
    mapState.altLines.push(L.polyline(alt.coords.map((c) => [c.lat, c.lon]), { color: '#6b7a99', weight: 3, dashArray: '6 6', opacity: 0.7 }).addTo(mapState.map));
  }
  mapState.line = L.polyline(r.coords.map((c) => [c.lat, c.lon]), { color: '#22c55e', weight: 5 }).addTo(mapState.map);
  mapState.dest = L.marker([Route.state.dest.lat, Route.state.dest.lon]).addTo(mapState.map);
  mapState.map.fitBounds(mapState.line.getBounds(), { padding: [20, 20] });
}

async function requestWakeLock() {
  try {
    if ('wakeLock' in navigator && !wakeLock) {
      wakeLock = await navigator.wakeLock.request('screen');
      wakeLock.addEventListener('release', () => { wakeLock = null; });
      log('화면 꺼짐 방지 켜짐 (경로 안내 중)');
    }
  } catch (e) {
    log(`화면 꺼짐 방지 실패: ${e.message}`);
  }
}

if (navUI.btnGeocode) {
  navUI.btnGeocode.addEventListener('click', async () => {
    const q = navUI.input.value.trim();
    if (!q) return;
    navUI.results.innerHTML = '<li class="muted">검색 중...</li>';
    try {
      const res = await Route.geocode(q);
      if (!res.length) { navUI.results.innerHTML = '<li class="muted">결과 없음</li>'; return; }
      navUI.results.innerHTML = res.map((r, i) => `<li data-i="${i}">${r.name}</li>`).join('');
      navUI.results.querySelectorAll('li').forEach((li) => li.addEventListener('click', () => {
        selectedDest = res[parseInt(li.dataset.i, 10)];
        navUI.results.querySelectorAll('li').forEach((x) => (x.style.borderColor = ''));
        li.style.borderColor = '#22c55e';
        navUI.btnStart.disabled = false;
        log(`도착지 선택: ${selectedDest.name}`);
      }));
    } catch (e) {
      navUI.results.innerHTML = `<li class="muted">검색 실패: ${e.message}</li>`;
    }
  });
  navUI.input.addEventListener('keydown', (e) => { if (e.key === 'Enter') navUI.btnGeocode.click(); });

  navUI.btnStart.addEventListener('click', async () => {
    if (!selectedDest) return;
    if (!Geo.state.pos) { log('✗ 현재 위치를 아직 모릅니다'); startGeo(); return; }
    navUI.status.textContent = '경로 탐색 중... (대안 경로와 위험물·인파 데이터 비교)';
    try {
      await Route.plan(Geo.state.pos, selectedDest, { considerHazards: true, crowd: crowdOptions() });
      const r = Route.state.chosen;
      log(`✓ 경로 ${fmtDist(r.distance)} · ${Math.round(r.duration / 60)}분 · 점수 ${Math.round(r.score.total)} (대안 ${Route.state.routes.length}개)`);
      const ci = r.score.crowdInfo;
      if (ci) log(`인파: 지수 ${ci.index}/km (${ci.time.label} ×${ci.time.factor}) · 가산 ${ci.penalty}초${ci.seoul.length ? ' · 서울 실시간 ' + ci.seoul.map((s) => `${s.name} ${s.level}`).join(', ') : ''}`);
      ensureMap();
      renderRoute();
      renderNavStatus();
      requestWakeLock();
      if (settings.voice) Route.speak(`경로 안내를 시작합니다. ${fmtDist(r.distance)}, 약 ${Math.round(r.duration / 60)}분`);
    } catch (e) {
      log(`✗ 경로 탐색 실패: ${e.message}`);
      navUI.status.textContent = `경로 탐색 실패: ${e.message}`;
    }
  });
  navUI.btnStop.addEventListener('click', () => {
    Route.stopSimulation();
    Route.stopNav();
    renderRoute();
    renderNavStatus();
    if (wakeLock) { wakeLock.release(); wakeLock = null; }
    log('경로 안내 중지');
  });
  navUI.btnSim.addEventListener('click', () => {
    if (!Route.state.active) { log('먼저 경로 안내를 시작하세요'); return; }
    Route.simulateWalk(5, 400);
    log('시뮬레이션: 경로를 따라 이동 (초당 약 12m, 실제 걸음의 약 8배)');
  });
}

Geo.onChange(() => {
  if (mapState.map && Geo.state.pos) {
    mapState.me.setLatLng([Geo.state.pos.lat, Geo.state.pos.lon]);
    renderHazards();
  }
  if (Route.state.active) Route.update(Geo.state.pos);
  renderNavStatus();
});

Route.on((type, payload) => {
  if (type === 'say') {
    log(`🔊 ${payload}`);
    if (settings.voice) Route.speak(payload);
  } else if (type === 'rerouted') {
    log(`↻ 경로 재탐색: ${fmtDist(payload.distance)}`);
    renderRoute();
  } else if (type === 'arrived') {
    renderRoute();
    if (wakeLock) { wakeLock.release(); wakeLock = null; }
  } else if (type === 'error') {
    log(`✗ 경로 오류: ${payload.message}`);
  }
});

// 디버그용으로 콘솔에서 접근
window.Geo = Geo;
window.Route = Route;
