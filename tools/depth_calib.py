"""metric depth 모델(미터 출력)로 샘플 사진의 전방 거리 통계를 계산한다.  사용: py -3 -I depth_calib_metric.py <model.onnx> [S]"""
import sys, os, time, numpy as np, onnxruntime as ort
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8")
M = sys.argv[1]
S = int(sys.argv[2]) if len(sys.argv) > 2 else 252
A = r"C:\Users\user\AppData\Local\Temp\claude\C--Users-user-Desktop-minjae-main\191d02a1-dab8-4201-977f-ba5737643f8d\scratchpad\pptx\assets"
G = r"C:\Users\user\Desktop\minjae-gps\samples"
sess = ort.InferenceSession(M, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]; out = sess.get_outputs()[0]
print("input", inp.name, inp.shape, "output", out.name, out.shape)
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)

def stats(path):
    im = Image.open(path).convert("RGB"); w, h = im.size; side = min(w, h)
    im = im.crop(((w - side) // 2, (h - side) // 2, (w - side) // 2 + side, (h - side) // 2 + side)).resize((S, S), Image.BILINEAR)
    x = ((np.asarray(im, np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None]
    t0 = time.time(); d = sess.run([out.name], {inp.name: x})[0]; ms = (time.time() - t0) * 1000
    d = d.reshape(d.shape[-2], d.shape[-1]); H, W = d.shape
    c = d[int(.25 * H):int(.75 * H), int(.3 * W):int(.7 * W)]
    floor = np.median(d[int(.9 * H):, int(.2 * W):int(.8 * W)])
    top = np.median(d[:int(.25 * H), int(.3 * W):int(.7 * W)])
    return dict(center_med=np.median(c), center_p20=np.percentile(c, 20), near_frac=(c < 1.5).mean(), floor=floor, top=top, dmin=d.min(), dmax=d.max(), ms=ms)

cases = [("street1 열린 길", os.path.join(G, "street1.jpg")), ("street2 열린 길", os.path.join(G, "street2.jpg")),
         ("shot_5 복도 열림", os.path.join(A, "shot_5.jpg")), ("shot_4 계단 위", os.path.join(A, "shot_4.jpg")),
         ("shot_3 계단 아래", os.path.join(A, "shot_3.jpg")), ("image-10-1 닫힌 문 막힘", os.path.join(A, "image-10-1.png")),
         ("image-3-1 횡단보도", os.path.join(A, "image-3-1.jpg")), ("app_shot 길", os.path.join(A, "app_shot.png"))]
for name, p in cases:
    s = stats(p)
    print(f"{name:22s} center {s['center_med']:.2f}m (p20 {s['center_p20']:.2f}) near<1.5m {s['near_frac']*100:3.0f}%  floor {s['floor']:.2f}  top {s['top']:.2f}  range {s['dmin']:.2f}~{s['dmax']:.2f}  {s['ms']:.0f}ms")
