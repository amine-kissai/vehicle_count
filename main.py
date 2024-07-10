import math

from ultralytics import YOLO, solutions
import cv2

from utils import *
from constants import *


def main():
    # Creating cam object
    cap = cv2.VideoCapture("videos/cars.mp4")

    # Setting up the YOLO model
    model = YOLO("yolo_weights/yolov8l.pt")

    # Setting up the mask
    mask = cv2.imread("mask.png")

    # counter = solutions.ObjectCounter(
    #     view_img=True,
    #     reg_pts=line_limits,
    #     classes_names=model.names,
    #     draw_tracks=True,
    #     line_thickness=2,
    # )

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read frame. Exiting...")
            break
        imgRegion = cv2.bitwise_and(img, mask)
        # Setting up the line for the counting
        cv2.line(img, line_limits[0], line_limits[1], COLORRED, 5)


        results = detect_vehicles(model, imgRegion)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = list(map(int, box.xyxy[0]))
                w, h = x2 - x1, y2 - y1
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                draw_rect(img=img, boundaries=(x1, y1, w, h), conf=conf, cls=model.names[cls], color=COLORBLACK)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()