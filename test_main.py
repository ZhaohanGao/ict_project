import cv2
import time
import uuid
from detector import load_model, detect_vehicles, estimate_speed_by_length
from license import extract_vehicle_features

VIDEO_PATH = "data/test2.mp4"
SHOW = False

net = load_model()
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30

trackers = {}
track_data = {}
frame_id = 0

print("\n🚗 processing...\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1

    if frame_id % 50 == 1:
        detections = detect_vehicles(frame, net)
        for det in detections:
            tracker = cv2.TrackerCSRT_create()
            x, y, w, h = det['bbox']
            h_frame, w_frame = frame.shape[:2]
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                continue
            if x + w > w_frame or y + h > h_frame:
                continue

            tracker.init(frame, (x, y, w, h))
            car_id = str(uuid.uuid4())[:8]
            trackers[car_id] = tracker
            track_data[car_id] = {
                "positions": [(frame_id, (x + w//2, y + h))],
                "bboxes": [(x, y, w, h)],
                "class": det['class_name'],
                "features": extract_vehicle_features(frame, det['bbox'], det['class_name'])
            }

    delete_ids = []
    for car_id, tracker in trackers.items():
        success, box = tracker.update(frame)
        if not success:
            delete_ids.append(car_id)
            continue

        x, y, w, h = map(int, box)
        cx, cy = x + w//2, y + h
        track_data[car_id]["positions"].append((frame_id, (cx, cy)))
        track_data[car_id]["bboxes"].append((x, y, w, h))
        track_data[car_id]["bbox"] = (x, y, w, h)
        track_data[car_id]["speed"] = estimate_speed_by_length(
            track_data[car_id]["positions"],
            track_data[car_id]["bboxes"],
            fps,
            track_data[car_id]["class"]
        )

    for car_id in delete_ids:
        trackers.pop(car_id)

cap.release()
cv2.destroyAllWindows()

for car_id, info in track_data.items():
    print({
        "id": car_id,
        "type": info['features']['type'],
        "color": info['features']['dominant_color'],
        "plate": info['features'].get('plate'),
        "speed_kmh": info.get("speed", 0.0)
    })
