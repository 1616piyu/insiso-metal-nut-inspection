
from ultralytics import YOLO

model = YOLO(
    "runs/detect/runs/metal_nut/improved_v2/weights/best.onnx"
)

results = model.val(
    data="configs/metal_nut.yaml",
    split="test",
    imgsz=256,
    batch=1,
    workers=0,
    plots=True,
    project="runs/metal_nut",
    name="improved_v2_256_test",
    exist_ok=True
)

print("\n========== 256x256 TEST RESULTS ==========")
print("Precision:", results.box.mp)
print("Recall:", results.box.mr)
print("mAP50:", results.box.map50)
print("mAP50-95:", results.box.map)
print("==========================================")
