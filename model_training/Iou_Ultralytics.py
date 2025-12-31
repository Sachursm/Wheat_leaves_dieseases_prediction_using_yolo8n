from ultralytics import YOLO

model = YOLO("model_training/runs/detect/train/weights/best.pt")
results = model.val(data="model_training/wheat.yaml")

print("Precision:", results.box.mp)
print("Recall   :", results.box.mr)
print("mAP@0.5  :", results.box.map50)      # IoU >= 0.5
print("mAP@0.5:0.95 :", results.box.map)    # IoU averaged (0.5 → 0.95)
