from ultralytics import YOLO
import cv2
import os

# Load YOLOv8 model (auto-downloads if missing)
model = YOLO("yolov8n.pt")

def detect_image(image_path: str):
    results = model(image_path, conf=0.4)

    img = cv2.imread(image_path)
    detections = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = model.names[cls_id]

            detections.append({
                "label": label,
                "confidence": round(conf, 2),
                "box": [x1, y1, x2, y2]
            })

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    os.makedirs("results", exist_ok=True)
    output_path = f"results/{os.path.basename(image_path)}"
    cv2.imwrite(output_path, img)

    return detections, output_path
