print("Script started.")
from ultralytics import YOLO
print("YOLO imported successfully.")

AUGMENTATION = dict(
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.4,
    fliplr=0.5,
    flipud=0.0,
    mosaic=0.2,
    scale=0.4,
    translate=0.15,
    degrees=7.0,
    mixup=0.1,
    dropout=0.15,
)


def main():
    print("Loading model.")
    model = YOLO('yolov8n.pt')
    print("Model has loaded successfully.")

    # --- Phase 1: Frozen backbone (feature extraction) ---
    # Freeze layers 0-9 (the backbone) to preserve COCO object detection weights.
    # Only the neck and head will train on your quickdraw/climbing data.
    print("Starting Phase 1 training (frozen backbone).")
    results = model.train(
        data='data.yaml',
        epochs=12, 
        imgsz=640,
        batch=16,
        patience=7,
        freeze=10,
        lr0=1e-3, 
        save=True,
        val=True,
        save_period=10,
        plots=True,
        weight_decay=0.001,
        name='Deck_Detector_Phase1_FrozenBackbone',
        project='runs/feature_extraction',
    )
    print("Phase 1 complete. Best model saved at:", results.save_dir)

    # --- Phase 2: Full fine-tuning ---
    # Load the best checkpoint from Phase 1, then unfreeze everything.
    # Use a low LR to avoid overwriting the backbone's pretrained weights.
    best_checkpoint = str(results.save_dir) + '/weights/best.pt'
    print(f"Loading Phase 1 best checkpoint: {best_checkpoint}")
    model = YOLO(best_checkpoint)

    print("Starting Phase 2 training (full fine-tuning).")
    results = model.train(
        data='data.yaml',
        epochs=50,              # Remaining budget
        imgsz=640,
        batch=16,
        patience=7,
        freeze=0,                # Unfreeze all layers
        lr0=1e-4,                # Lower LR — fine-tuning, not from scratch
        save=True,
        val=True,
        save_period=10,
        plots=True,
        name='Deck_Detector_Phase2_FullFinetune',
        project='runs/feature_extraction',
        weight_decay=0.001,
    )
    print("Phase 2 complete. Best model saved at:", results.save_dir)


if __name__ == "__main__":
    main()