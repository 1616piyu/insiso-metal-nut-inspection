
from ultralytics import YOLO

model = YOLO(
    "runs/detect/runs/metal_nut/improved_v2/weights/best.pt"
)

model.export(
    format="onnx",
    imgsz=320,
    batch=1,
    dynamic=False,
    simplify=True,
    opset=17
)

print("320x320 ONNX export completed.")
