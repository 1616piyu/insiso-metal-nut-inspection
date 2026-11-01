
from ultralytics import YOLO

model = YOLO(
    "runs/detect/runs/metal_nut/improved_v2/weights/best.pt"
)

model.export(
    format="onnx",
    imgsz=256,
    batch=1,
    dynamic=False,
    simplify=True,
    opset=17
)

print("Improved_v2 256x256 ONNX export completed.")
