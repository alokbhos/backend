import os
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "civic_yolo.pt")

CONF_THRESHOLD = 0.75
VALID_CLASSES = ["garbage", "pothole", "broken_streetlight"]

model = YOLO(MODEL_PATH)

def detect_image(image_path: str):
    results = model.predict(
        source=image_path,
        conf=CONF_THRESHOLD,
        save=True,
        project="results",
        name="predict"
    )

    detections = []
    output_image = None

    for r in results:
        if r.save_dir:
            output_image = os.path.join(r.save_dir, os.path.basename(image_path))

        for box in r.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = model.names[cls_id]

            if label not in VALID_CLASSES:
                continue

            if confidence < CONF_THRESHOLD:
                continue

            detections.append({
                "label": label,
                "confidence": round(confidence * 100, 2),
                "bbox": box.xyxy[0].tolist()
            })

    if not detections:
        return [], None

    return detections, output_image
