from ultralytics import YOLO
from PIL import Image
import os

MODEL_PATH = "models/civic_yolo.pt"
UPLOAD_DIR = "uploads"

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

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = model.names[cls_id]

            # 🚫 Reject non-civic issues
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
        return {
            "status": "rejected",
            "reason": "No valid civic issue detected"
        }

    return {
        "status": "success",
        "detections": detections
    }
