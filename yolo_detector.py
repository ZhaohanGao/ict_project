# yolo_detector.py
from ultralytics import YOLO
import torch
import cv2

VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle'}

def load_model():
    model = YOLO("yolov8n.pt")  # 可选 yolov8s.pt/m.pt/l.pt
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"✅ YOLOv8 loaded on {device.upper()}")
    return model

def detect_vehicles(frame, model, threshold=0.5):
    results = model(frame)[0]
    detections = []

    for box, cls_id, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        class_name = model.names[int(cls_id)]
        if class_name in VEHICLE_CLASSES and conf >= threshold:
            x1, y1, x2, y2 = map(int, box.tolist())
            detections.append({
                "bbox": (x1, y1, x2 - x1, y2 - y1),
                "class_name": class_name,
                "confidence": float(conf)
            })
    return detections
