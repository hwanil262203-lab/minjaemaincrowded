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
};

// 클래스별 한국어 이름, 위험 가중치(0~1), 색. 키는 labels.json 의 영어 이름.
const CLASS_INFO = {
  crosswalk: { ko: '횡단보도', weight: 0.8, color: '#facc15' },
  stairs: { ko: '계단', weight: 1.0, color: '#f97316' },
  pole: { ko: '기둥·가로등', weight: 1.0, color: '#ef4444' },
  person: { ko: '사람', weight: 0.6, color: '#38bdf8' },
  vehicle: { ko: '차량', weight: 0.9, color: '#a855f7' },
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
async function loadModel() {
  try {
    setStatus('모델 로딩 중...');
    ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${CONFIG.ortVersion}/dist/`;

    // labels.json (없어도 동작: 클래스 이름이 숫자로 표시됨)
    try {
      const r = await fetch(CONFIG.labelsUrl, { cache: 'no-store' });
      if (r.ok) labels = { ...labels, ...(await r.json()) };
    } catch (_) {
      /* 무시 */
    }
    // 모델 파일이 없으면(아직 학습 전) 시험용 사전학습 모델로 대체
    let modelUrl = CONFIG.modelUrl;
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
    log(`모델 파일: ${modelUrl}`);

    // WebGPU 우선, 실패하면 WASM
    const providers = navigator.gpu ? ['webgpu', 'wasm'] : ['wasm'];
    let lastErr = null;
    for (const ep of providers) {
      try {
        session = await ort.InferenceSession.create(modelUrl, {
          executionProviders: [ep],
          graphOptimizationLevel: 'all',
        });
        engineName = ep;
        break;
      } catch (e) {
        lastErr = e;
        log(`${ep} 초기화 실패 → 다음 백엔드 시도 (${e.message.split('\n')[0]})`);
      }
    }
    if (!session) throw lastErr || new Error('세션 생성 실패');

    inputName = session.inputNames[0];
    outputName = session.outputNames[0];

    // 입력 크기: 모델 메타데이터 → labels.json → 640 순서로 결정
    inputSize = labels.imgsz || 640;
    const meta = session.inputMetadata && session.inputMetadata[0];
    const dims = meta && (meta.shape || meta.dims);
    if (dims && typeof dims[2] === 'number' && dims[2] > 0) inputSize = dims[2];

    // 워밍업 겸 출력 형식 확인
    const zeros = new ort.Tensor('float32', new Float32Array(3 * inputSize * inputSize), [1, 3, inputSize, inputSize]);
    const t0 = performance.now();
    const out = await session.run({ [inputName]: zeros });
    const od = out[outputName].dims;
    const nmsInModel = od.length === 3 && od[2] === 6;
    labels.nms_in_model = nmsInModel;

    ui.engine.textContent = `${engineName.toUpperCase()} · ${inputSize}px · NMS ${nmsInModel ? '내장' : 'JS'}` +
      (usingFallbackModel ? ' · 시험용 모델' : '');
    log(`✓ 모델 로드 완료 (${engineName}, 입력 ${inputSize}px, 출력 [${od.join(',')}], 워밍업 ${Math.round(performance.now() - t0)}ms)`);
    log(`클래스: ${labels.names.join(', ') || '(labels.json 없음)'}`);
    setStatus(usingFallbackModel ? '시험용 모델 준비 완료' : '모델 준비 완료');
  } catch (e) {
    log(`✗ 모델 로드 실패: ${e.message}`);
    console.error('모델 로드 오류:', e);
    setStatus('모델 로드 실패');
    session = null;
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
};

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
    ctx.strokeRect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);
    const text = `${det.ko} ${(det.score * 100).toFixed(0)}%`;
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
        `<b>${d.ko}</b> ${(d.score * 100).toFixed(0)}% · ${DIR_KO[d.dir]} · 근접 ${(d.prox * 100).toFixed(0)}%</li>`;
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

async function inferenceLoop() {
  if (!session || !(webcamStream || testImage)) {
    log('✗ 모델 또는 카메라가 준비되지 않았습니다.');
    return;
  }
  if (isRunning) return;
  isRunning = true;
  setStatus('감지 중...');
  log('감지 시작됨');
  const ctx = ui.overlay.getContext('2d');

  const step = async () => {
    if (!isRunning) return;
    const t0 = performance.now();
    try {
      const src = frameSource();
      const vw = src.w;
      const vh = src.h;
      if (vw && vh) {
        if (ui.overlay.width !== vw) {
          ui.overlay.width = vw;
          ui.overlay.height = vh;
        }
        const conf = parseFloat(ui.threshold.value) || 0.25;
        const requiredFrames = parseInt(ui.steadyFrames.value, 10) || 3;
        const minProximity = parseFloat(ui.proximity.value) || 0.2;

        const lb = preprocess(src, inputSize);
        const result = await session.run({ [inputName]: lb.tensor });
        const dets = postprocess(result[outputName], lb, conf, vw, vh);
        const best = assess(dets, vw, vh, minProximity);
        const confirmed = updateHysteresis(best ? best.label : null, requiredFrames);

        drawDetections(ctx, dets, best);
        updateDetectionList(dets, best);
        updateFloatingIndicator(!!confirmed);

        if (confirmed && best) vibrateFor(best);
        else stopVibration();
        health.lastFrameTs = performance.now();
      }
    } catch (e) {
      console.error('추론 오류:', e);
      log(`감지 오류: ${e.message}`);
      stopLoop();
      return;
    }
    // 성능 표시
    perf.ms = perf.ms * 0.8 + (performance.now() - t0) * 0.2;
    perf.frames += 1;
    if (performance.now() - perf.lastReport > 1000) {
      perf.lastReport = performance.now();
      setStatus(`감지 중 · ${Math.round(perf.ms)}ms/프레임 (${(1000 / Math.max(1, perf.ms)).toFixed(1)} FPS)`);
    }
    animationHandle = requestAnimationFrame(step);
  };
  step();
}

function stopLoop() {
  isRunning = false;
  stopVibration();
  updateFloatingIndicator(false);
  setStatus('정지됨');
  if (animationHandle) cancelAnimationFrame(animationHandle);
}

ui.btnCamera.addEventListener('click', openCamera);
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

  const imageUrl = new URLSearchParams(location.search).get('image');
  if (imageUrl) {
    if (!testImage) await loadTestImage(imageUrl);
  } else if (!webcamStream) {
    await openCamera();
  }
  if (session && (webcamStream || testImage)) {
    log('🚀 감지를 시작합니다...');
    inferenceLoop();
  } else {
    log('✗ 시작 실패: 모델 또는 카메라가 준비되지 않았습니다.');
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
  await loadModel();
});
