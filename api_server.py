from datetime import datetime
import cv2
import uuid
import csv
import os
from flask import Flask, request, jsonify, send_file
from gtts import gTTS
from yolo_tracker import YOLOByteTrackWrapper, estimate_speed_by_length
from license import extract_vehicle_features
import random

app = Flask(__name__, static_folder=".", static_url_path="/")
tracker = YOLOByteTrackWrapper()
print(f"[DEBUG] Loaded model type: {type(tracker.model)}")
SPEED_LIMIT = 60.0  # km/h

def generate_bd_license_plate():
    city = "DHAKA"
    vehicle_type = "GA"
    year = str(random.randint(20, 24))
    number = str(random.randint(1, 9999)).zfill(4)
    return f"{city} {vehicle_type} {year}-{number}"

@app.route("/detect", methods=["POST"])
def violation_detect():
    camera_id = request.form.get("camera_id")
    if not camera_id:
        return jsonify({"error": "Missing camera_id"}), 400
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    camera_info = {}
    camera_file_path = os.path.join("speed_monitor_dashboard", "data", "cameras.csv")
    if os.path.isfile(camera_file_path):
        with open(camera_file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["camera_id"] == camera_id:
                    camera_info = row
                    break
    else:
        return jsonify({"error": "cameras.csv not found"}), 500

    latitude = camera_info.get("latitude", "")
    longitude = camera_info.get("longitude", "")
    speed_limit = float(camera_info.get("speed_limit", SPEED_LIMIT))

    video_file = request.files['video']
    os.makedirs("uploads", exist_ok=True)
    video_path = os.path.join("uploads", video_file.filename)
    video_file.save(video_path)

    cap = cv2.VideoCapture(video_path)
    output_path = os.path.join("uploads", "annotated_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS) or 30, (int(cap.get(3)), int(cap.get(4))))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    snapshot_dir = os.path.join("uploads", "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)

    track_data = {}
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        tracked_vehicles = tracker.detect_and_track(frame)
        for vehicle in tracked_vehicles:
            x, y, w, h = vehicle["bbox"]
            track_id = vehicle["id"]
            class_name = vehicle["class_name"]

            if track_id not in track_data:
                track_data[track_id] = {
                    "positions": [], "bboxes": [], "class": class_name,
                    "features": extract_vehicle_features(frame, (x, y, w, h), class_name),
                    "snapshot_frame": None
                }

            cx, cy = x + w // 2, y + h
            track_data[track_id]["positions"].append((frame_id, (cx, cy)))
            track_data[track_id]["bboxes"].append((x, y, w, h))
            track_data[track_id]["bbox"] = (x, y, w, h)
            track_data[track_id]["snapshot_frame"] = frame.copy()
            track_data[track_id]["speed"] = estimate_speed_by_length(
                track_data[track_id]["positions"],
                track_data[track_id]["bboxes"],
                fps,
                class_name
            )

            color = (0, 255, 0) if track_data[track_id]['speed'] <= SPEED_LIMIT else (0, 0, 255)
            label = f"{class_name} {track_data[track_id]['speed']:.1f} km/h ID:{track_id}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        out_writer.write(frame)

    cap.release()
    out_writer.release()

    overspeed_vehicles = []
    db_path = os.path.join("speed_monitor_dashboard", "data", "incidents.csv")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    file_exists = os.path.isfile(db_path)

    if file_exists:
        with open(db_path, "rb+") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b'\n':
                f.write(b'\n')

    with open(db_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "timestamp", "camera_id", "license_plate", "latitude", "longitude",
            "speed_limit", "actual_speed", "speed_difference", "image_url"
        ])
        if not file_exists:
            writer.writeheader()

        for car_id, info in track_data.items():
            speed = info.get("speed", 0.0)
            if speed > speed_limit:
                snapshot_path = os.path.join(snapshot_dir, f"{car_id}.jpg")
                snapshot = info.get("snapshot_frame")
                if snapshot is not None:
                    x, y, w, h = info.get("bbox", (0, 0, 0, 0))
                    cv2.rectangle(snapshot, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    label = f"{speed:.1f} km/h"
                    plate = info["features"].get("plate", car_id)
                    cv2.putText(snapshot, f"{label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(snapshot, f"Plate: {plate}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv2.imwrite(snapshot_path, snapshot)

                row = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "camera_id": camera_id,
                    "license_plate": plate,
                    "latitude": latitude,
                    "longitude": longitude,
                    "speed_limit": speed_limit,
                    "actual_speed": speed,
                    "speed_difference": speed - speed_limit,
                    "image_url": f"../uploads/snapshots/{car_id}.jpg"
                }
                writer.writerow(row)
                overspeed_vehicles.append(row)

    if overspeed_vehicles:
        lines = [
            f"Vehicle {v['license_plate']} is overspeeding at {v['actual_speed']} kilometers per hour."
            for v in overspeed_vehicles
        ]
        tts_text = " ".join(lines)
    else:
        tts_text = "No overspeeding vehicles detected."

    tts = gTTS(tts_text)
    audio_path = os.path.join("uploads", "overspeed_alert.mp3")
    tts.save(audio_path)

    # 返回音频文件作为流媒体（不返回 JSON）
    return send_file(
        audio_path,
        mimetype="audio/mpeg",
        as_attachment=False,
        download_name="overspeed_alert.mp3"
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)
