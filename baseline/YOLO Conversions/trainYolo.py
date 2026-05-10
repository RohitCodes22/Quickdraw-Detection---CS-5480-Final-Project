import json
import os
from pathlib import Path

classes = ["quickdraw"]

json_dir = "train/jsons"
output_dir = "train/labels"

os.makedirs(output_dir, exist_ok=True)

def convert_bbox(img_w, img_h, points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]

    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    x_center = (x_min + x_max) / 2.0 / img_w
    y_center = (y_min + y_max) / 2.0 / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h

    return x_center, y_center, width, height

for json_file in os.listdir(json_dir):
    if not json_file.endswith(".json"):
        continue

    json_path = os.path.join(json_dir, json_file)

    with open(json_path, "r") as f:
        data = json.load(f)

    img_w = data["imageWidth"]
    img_h = data["imageHeight"]

    label_file = os.path.join(output_dir, json_file.replace(".json", ".txt"))

    with open(label_file, "w") as out:
        for shape in data["shapes"]:
            label = shape["label"]

            if label not in classes:
                print(f"Skipping unknown class: {label}")
                continue

            class_id = classes.index(label)
            points = shape["points"]

            bbox = convert_bbox(img_w, img_h, points)

            out.write(f"{class_id} {' '.join(map(str, bbox))}\n")

print("Conversion complete.")