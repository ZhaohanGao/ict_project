import numpy as np
from tracker.byte_tracker import BYTETracker
from tracker.byte_tracker import STrack
from ultralytics import YOLO
import torch

VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}

DEFAULT_VEHICLE_LENGTHS = {
    'car': 4.5,
    'truck': 12.0,
    'bus': 10.0,
    'motorcycle': 2.0
}

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

class YOLOByteTrackWrapper:
    def __init__(self, model_path="yolov8n.pt", threshold=0.5, frame_rate=30):
        self.model = YOLO(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"✅ YOLOv8 loaded on {self.device.upper()}")
        self.threshold = threshold
        self.byte_tracker = BYTETracker(frame_rate=frame_rate)

    def detect_and_track(self, frame):
        result = self.model(frame, verbose=False)[0]
        detections = []

        for box, cls_id, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            class_name = self.model.names[int(cls_id)]
            if class_name in VEHICLE_CLASSES and conf > self.threshold:
                x1, y1, x2, y2 = map(int, box.tolist())
                detections.append([x1, y1, x2, y2, conf.item(), int(cls_id)])

        tracks = []
        if detections:
            dets_for_byte = np.array(detections)
            online_targets = self.byte_tracker.update(dets_for_byte, frame.shape[:2], frame.shape[:2])

            for t in online_targets:
                x, y, w, h = map(int, t.tlwh)
                track_id = t.track_id
                class_name = self.model.names[t.class_id]
                tracks.append({
                    "id": track_id,
                    "bbox": (x, y, w, h),
                    "class_name": class_name
                })

        return tracks
