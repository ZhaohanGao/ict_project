# 🚗 ICT project

This is a Flask-based backend for detecting vehicles in video, estimating their speeds, identifying overspeeding events, and generating audio alerts.

## Features

- Detects cars, trucks, buses, motorcycles using a pre-trained detector
- Estimates speed based on bounding box motion and known vehicle dimensions
- Flags overspeeding vehicles and stores data in `uploads/database.csv`
- Returns TTS audio feedback for overspeeding results
- Speed limit is configurable via API

## Installation

```bash
pip install -r requirements.txt
python api_server.py
```

## Endpoints

### `POST /detect`

Analyze a video file and return a generated MP3 with overspeeding announcements.

```bash
curl -X POST http://localhost:5000/detect -F "video=@your_video.mp4" --output alert.mp3
```

### `GET /vehicles`

Returns all recorded vehicle data from the CSV database.

### `GET /violations`

Returns vehicles exceeding the current speed limit.

### `GET /get_speed_limit`

Returns the current speed limit in km/h.

### `POST /set_speed_limit`

Update the speed limit.

```bash
curl -X POST http://localhost:5000/set_speed_limit \
  -H "Content-Type: application/json" \
  -d '{"value": 50}'
```

## Notes

- EasyOCR is used for license plate detection (only English supported).
- TTS is powered by gTTS (Google Text-to-Speech).
- All results are stored under the `uploads/` directory.
