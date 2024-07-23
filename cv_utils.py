import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cvzone

from contextlib import contextmanager

from constants import classNames


def user_typed_q() -> bool:
    return (cv2.waitKey(1) & 0xFF) == ord("q")

def apply_mask(img:np.ndarray, mask:np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(img, mask)

def is_crossing_line(bbox, line_limits) -> bool:
    x1, y1, x2, y2 = bbox
    bbox_center_y = (y1 + y2) // 2
    line_y = line_limits[0][1]
    return y1 < line_y < y2 and bbox_center_y > line_y

def detect_vehicles(model, image_path, vehicle_classes = [2, 3, 5, 7] ):
    results = model(image_path, classes=vehicle_classes, stream=True)  # Pass the vehicle classes to filter
    return results

def track_vehicles(model, image, vehicle_classes=[2, 3, 5, 7]):
    results = model.track(source=image, classes=vehicle_classes, persist=True)  # Pass the vehicle classes to filter
    return results

### Context managers
@contextmanager
def video_capture(path:str):
    cap = cv2.VideoCapture(path)
    try:
        yield cap
        cv2.waitKey(1)
    finally:
        cap.release()
        cv2.destroyAllWindows()

@contextmanager
def destroy_all_window_on_exit():
    try:
        yield None
    finally:
        cv2.destroyAllWindows()

### YOLO utils
def train_yolo(model, data:str, epochs:int, imgsz:int)-> None:
    model.train(data=data, epochs=epochs, imgsz=imgsz)

def save_trained_yolo(model, directory:str):
    model.save(directory)


### Drawing utils
def draw_rect(img, boundaries:tuple, conf:float, cls:str, color:tuple) -> None:
    cvzone.cornerRect(img, boundaries, l=9, rt=2, colorR=color)
    cvzone.putTextRect(img, f"{conf} {cls}", (max(0, boundaries[0]), max(20, boundaries[1])), scale=1, thickness=1, offset=2, colorR=color)

def display_vehicle_count(img, vehicle_count, class_names, color):
    y_offset = 30
    for cls, count in vehicle_count.items():
        cvzone.putTextRect(img, f"{class_names[cls]}: {count}", (30, y_offset), scale=1, thickness=1, offset=2, colorR=color)
        y_offset += 30
