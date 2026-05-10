from ultralytics import YOLO
model = YOLO('yolov8n.pt')

model.train(
    data = 'data.yaml',
    epochs = 100,
    patience = 10,
    imgsz = 640,
    batch = 16,
    name = 'baseline_model',
    project = 'runs/baseline',
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    degrees=0.0,
    translate=0.0,
    scale=0.0,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.0,
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0
)

metrics = model.val(split = "test")
print(metrics)

