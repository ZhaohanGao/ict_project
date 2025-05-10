# ict_project

# Project Structure
| Module Category | Name | Description |
|----------------|------|-------------|
| 🎥 Input & Detection | `detector.py` | Vehicle detection and tracking | 做一个简单的交通工具检测，识别检测目标
| 🚗 Speed Logic | `speed_estimator.py` | Speed computation from displacement |
| 🔍 OCR | `license_plate.py` | License plate recognition |
| ⚠️ Overspeed Logic | `violation_checker.py` | Detect overspeeding and package records |
| 🔊 Feedback | `feedback.py` | Play voice alerts | 声音合成相关方法
| 📁 Logging | `logger.py` | Write CSV/JSON logs(better to use mysql) |
| 🌐 Backend Server | `app.py` (Flask) | Expose REST API for stats, config, upload |
| 📊 Dashboard | `dashboard/` | Local or mock interface for data summaries |


### POST /upload_frame image/video uploaded for analysis, response: {plate, violation status(true of false), speed, limit),， 同时在接口处合成播放声音车牌xxx，车速xxx，已违规，声音由后端合成返回前端，最好能做一个环境识别，记录高频违规区域，存储数据便于分析
def estimate_speed(displacement: Tuple[int, int], fps: float, pixel_to_meter: float) -> float 速度估算
def check_violation(vehicle_id: int, speed: float, speed_limit: float) -> bool 超速检测
def generate_violation_record(vehicle_id: int, plate: str, speed: float, timestamp: str) -> Dict
def recognize_plate(image: np.ndarray) -> str 车牌识别

### GET /violations get violation records
### Get /violations?plate= get violation record for certain plate
一些数据操作相关接口

### POST /set_speed_limit
def set_speed_limit(zone: str) -> float
