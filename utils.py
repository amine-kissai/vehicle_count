from ultralytics import YOLO
import cvzone

import yaml
from typing import List

def train_yolo(model, data:str, epochs:int, imgsz:int)-> None:
    model.train(data=data, epochs=epochs, imgsz=imgsz)

def save_trained_yolo(model, directory:str):
    model.save(directory)

def make_yaml(train_path:str, val_path:str, nc:int, names_array: List[str]) -> None:
    data_yaml = dict(
    train = train_path,
    val = val_path,
    nc = nc,
    names = names_array
    )
    # Note that I am creating the file in the yolov5/data/ directory.
    with open('data.yaml', 'w') as outfile:
        yaml.dump(data_yaml, outfile, default_flow_style=True)
    
def draw_rect(img, boundaries:tuple, conf:float, cls:str, color:tuple) -> None:
    cvzone.cornerRect(img, boundaries, l=9, rt=2, colorR=color)
    cvzone.putTextRect(img, f"{conf} {cls}", (max(0, boundaries[0]), max(20, boundaries[1])), scale=1, thickness=1, offset=2, colorR=color)

def detect_vehicles(model, image_path, vehicle_classes = [2, 3, 5, 7] ):
    results = model(image_path, classes=vehicle_classes, stream=True)  # Pass the vehicle classes to filter
    return results

def track_vehicles(model, image, vehicle_classes=[2, 3, 5, 7]):
    results = model.track(source=image, classes=vehicle_classes, persist=True)  # Pass the vehicle classes to filter
    return results


def is_crossing_line(bbox, line_limits):
    # Implement logic to check if the bbox is crossing the line_limits
    # Example: Check if the center of the bbox is crossing the line
    x1, y1, x2, y2 = bbox
    bbox_center_y = (y1 + y2) // 2
    line_y = line_limits[0][1]
    return y1 < line_y < y2 and bbox_center_y > line_y

def display_vehicle_count(img, vehicle_count, class_names, color):
    y_offset = 30
    for cls, count in vehicle_count.items():
        cvzone.putTextRect(img, f"{class_names[cls]}: {count}", (30, y_offset), scale=1, thickness=1, offset=2, colorR=color)
        y_offset += 30
