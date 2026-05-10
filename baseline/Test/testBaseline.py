from ultralytics import YOLO
import os

model = YOLO("C:\\Users\\rohit\\runs\\detect\\runs\\baseline\\baseline_model\\weights\\best.pt")

metrics = model.val(
    data="data.yaml",
    split="test",
    imgsz=640,
    batch=16,
    name="quickdraw_baseline_test",
    project="runs/detect",
)

print("=== Test Set Metrics ===")
print(f"mAP50:        {metrics.box.map50:.4f}")
print(f"mAP50-95:     {metrics.box.map:.4f}")
print(f"Precision:    {metrics.box.mp:.4f}")
print(f"Recall:       {metrics.box.mr:.4f}")

test_images_dir = "test/images"
 
supported_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")
image_paths = [
    os.path.join(test_images_dir, f)
    for f in os.listdir(test_images_dir)
    if f.lower().endswith(supported_ext)
]

model.predict(
    source=test_images_dir,   
    imgsz=640,
    conf=0.25,          
    iou=0.45,               
    save=True,              
    save_txt=True,         
    save_conf=True,         
    name="quickdraw_baseline_preds",
    project="runs/detect",
)

print("\nPrediction images saved to runs/detect/quickdraw_baseline_preds/")