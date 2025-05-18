from ultralytics import YOLO
import torch
import cv2

VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle'}

DEFAULT_VEHICLE_LENGTHS = {
    'car': 4.5,
    'truck': 12.0,
    'bus': 10.0,
    'motorcycle': 2.0
}

def load_model():
    model = YOLO("yolov8n.pt")  # 可替换为 yolov8s.pt/yolov8m.pt 等更大模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"✅ YOLOv8 loaded on {device.upper()}")
    return model

def detect_vehicles(frame, model, threshold=0.5):
    # YOLO 可以直接接受 OpenCV 图像（BGR）
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

def estimate_speed_by_length(position_history, bbox_history, fps, vehicle_class):
    if len(position_history) < 2:
        return 0.0
    (f1, p1), (f2, p2) = position_history[0], position_history[-1]
    dt = (f2 - f1) / fps
    if dt == 0:
        return 0.0

    real_length = DEFAULT_VEHICLE_LENGTHS.get(vehicle_class, 4.0)
    pixel_heights = [bbox[3] for bbox in bbox_history[-3:]]
    avg_pixel_height = sum(pixel_heights) / len(pixel_heights)
    if avg_pixel_height == 0:
        return 0.0

    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    pixel_dist = (dx ** 2 + dy ** 2) ** 0.5
    meters_per_pixel = real_length / avg_pixel_height
    dist_m = pixel_dist * meters_per_pixel
    speed = dist_m / dt * 3.6
    return round(speed, 1)
