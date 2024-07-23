# Vehicle Counting using YOLOv8

This project implements vehicle counting using the YOLOv8 object detection model. The script processes a video file to count vehicles that cross a specified line.

## Features

- Counts vehicles from a video file.
- Applies a mask to focus on a specific region of the video.
- Displays vehicle counts on the video frames.
- Customizable line coordinates for counting.

## Requirements

- Python 3.x
- OpenCV
- Ultralytics YOLO
- cvzone
- argparse

## Installation

1. Clone this repository.
2. Install the required Python packages:

```sh
pip install opencv-python ultralytics cvzone argparse
```

# Usage
## Command-Line Arguments
--video: Path to the video file (required).
--mask: Path to the mask image (required).
--line: Coordinates of the counting line in the format x1 y1 x2 y2 (required).

# Example

To run the script:
```sh
python main.py --video "videos/cars.mp4" --mask "masks/mask.png" --line 400 297 673 297
```
Or you can use directly the example.py script
```sh
python example.py
```

# Notes

- Make sure the mask image and video resolution match to avoid erros
- The line coordinates should be chosen carefully based on the video perspective to ensure accurate counting.
- The videos for this project were downloaded from cvzone and pexel.com
- Some of the masks were generated using canva.com