import cv2
import numpy as np
import easyocr

import torch
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())


def extract_vehicle_features(frame, bbox, class_name):
    x, y, w, h = bbox
    crop = frame[y:y+h, x:x+w]
    if crop.size == 0:
        return {}

    vehicle_type = class_name
    center = (x + w // 2, y + h)
    avg_color = tuple(np.mean(crop.reshape(-1, 3), axis=0).astype(int)[::-1])

    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_mean = np.mean(hsv_crop[:, :, 0])
    dominant_color = interpret_hue(h_mean)

    try:
        result = reader.readtext(crop)
        plate = result[0][1] if result else ""
    except:
        plate = ""

    return {
        'type': vehicle_type,
        'center': center,
        'size': (w, h),
        'color_rgb': avg_color,
        'dominant_color': dominant_color,
        'plate': plate
    }

def interpret_hue(hue):
    if hue < 15 or hue >= 160:
        return "red"
    elif 15 <= hue < 35:
        return "yellow"
    elif 35 <= hue < 85:
        return "green"
    elif 85 <= hue < 125:
        return "blue"
    elif 125 <= hue < 160:
        return "purple"
    return "unknown"
