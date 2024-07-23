import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2

import math
from collections import defaultdict

from utils import plot_vehicle_pie_chart
from constants import COLORBLACK, COLORGREEN, COLORRED, line_limits2
from cv_utils import user_typed_q, apply_mask, track_vehicles, draw_rect, is_crossing_line, display_vehicle_count




def main():
    # Creating cam object
    cap = cv2.VideoCapture("videos/cars2.mp4")

    # Setting up the YOLO model
    model = YOLO("yolo_weights/yolov8l.pt")

    # Setting up the mask
    mask = cv2.imread("masks/mask2.png")

    vehicle_count = defaultdict(int)

    counted_ids = set()


    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Failed to read frame. Exiting...")
            break

        imgRegion = cv2.bitwise_and(img, mask)
        # Setting up the line for the counting
        cv2.line(img, line_limits2[0], line_limits2[1], COLORRED, 5)

        # Track vehicles
        results = track_vehicles(model, imgRegion)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = list(map(int, box.xyxy[0]))
                w, h = x2 - x1, y2 - y1
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                draw_rect(img=img, boundaries=(x1, y1, w, h), conf=conf, cls=model.names[cls], color=COLORBLACK)
                if box.id :
                    track_id = int(box.id[0])
                    if track_id not in counted_ids and is_crossing_line((x1, y1, x2, y2), line_limits2):
                        counted_ids.add(track_id)
                        vehicle_count[cls] += 1
                        cv2.line(img, line_limits2[0], line_limits2[1], COLORGREEN, 5)
        # Display vehicle counts
        display_vehicle_count(img, vehicle_count, model.names, color=COLORRED)

        cv2.imshow("Image", img)
        if user_typed_q():
            break

    cap.release()
    cv2.destroyAllWindows()

    plot_vehicle_pie_chart(vehicle_count)


if __name__ == "__main__":
    main()