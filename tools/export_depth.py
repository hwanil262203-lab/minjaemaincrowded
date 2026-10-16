"""Depth Anything V2 Metric Indoor Small 을 작은 고정 입력(SxS)으로 ONNX 내보내기 + fp16/int8 변환.
사용: py -3 export_depth.py 266"""
import sys, os, warnings
warnings.filterwarnings("ignore")
import torch, onnx
from transformers import AutoModelForDepthEstimation
from onnxruntime.transformers.onnx_model import OnnxModel
from onnxruntime.quantization import quantize_dynamic, QuantType
S = int(sys.argv[1]) if len(sys.argv) > 1 else 266
OUT = r"C:\Users\user\Desktop\minjae-gps\models"
name = f"depth_metric_indoor_small_{S}"
model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf").eval()

class Wrap(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, pixel_values): return self.m(pixel_values=pixel_values).predicted_depth

x = torch.zeros(1, 3, S, S)
fp32 = os.path.join(OUT, name + "_fp32.onnx")
torch.onnx.export(Wrap(model), x, fp32, input_names=["pixel_values"], output_names=["predicted_depth"], opset_version=17, dynamo=False)
m = onnx.load(fp32); onnx.checker.check_model(m)
om = OnnxModel(m); om.convert_float_to_float16(keep_io_types=True); om.save_model_to_file(os.path.join(OUT, name + "_fp16.onnx"))  # WebGPU 용 (입출력은 float32 유지)
quantize_dynamic(fp32, os.path.join(OUT, name + "_q8.onnx"), weight_type=QuantType.QUInt8)
os.remove(fp32)
for f in sorted(os.listdir(OUT)): print(f, round(os.path.getsize(os.path.join(OUT, f)) / 1e6, 1), "MB")
