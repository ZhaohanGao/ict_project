import torch
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle'}

DEFAULT_VEHICLE_LENGTHS = {
    'car': 4.5,
    'truck': 12.0,
    'bus': 10.0,
    'motorcycle': 2.0
}

def load_model():
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.eval()
    return model

def detect_vehicles(frame, model, threshold=0.5):
    image_tensor = F.to_tensor(frame).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)[0]

    results = []
    for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
        if score >= threshold:
            if label.item() >= len(COCO_INSTANCE_CATEGORY_NAMES):
                continue
            class_name = COCO_INSTANCE_CATEGORY_NAMES[label.item()]
            if class_name in VEHICLE_CLASSES:
                x1, y1, x2, y2 = box.int().tolist()
                results.append({
                    'bbox': (x1, y1, x2 - x1, y2 - y1),
                    'class_name': class_name
                })
    return results

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