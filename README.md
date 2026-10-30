# Vehicle Detection with YOLOv8

Detects vehicles (cars, motorcycles, buses, trucks) in a video using
[YOLOv8](https://github.com/ultralytics/ultralytics). A region-of-interest mask is
applied so only the road area is analysed.

## Features

- Vehicle detection with a pretrained `yolov8n` model
- Region-of-interest masking via `mask.png` (ignores sky, buildings, etc.)
- Bounding boxes with class name and confidence drawn on each frame

## Project Structure

```
.
├── car_detector.py    # Main script
├── mask.png           # 1280x720 ROI mask (white = analysed area)
├── 2.mp4              # Input video (add your own)
├── yolov8n.pt         # YOLOv8 weights (auto-downloaded on first run)
├── requirements.txt
└── README.md
```

## Installation

1. **Clone / download** this project and open a terminal in its folder.

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Add your video** and name it `2.mp4` (or change the filename in the script).

## Usage

```bash
python car_detector.py
```

Press **`q`** in the video window to quit.

## Configuration

| Setting | Location | Description |
|---|---|---|
| Video source | `cv2.VideoCapture("2.mp4")` | Path to input video, or `0` for webcam |
| Mask | `cv2.imread("mask.png")` | Must match the video resolution (1280x720) |
| Model | `YOLO('yolov8n.pt')` | Swap for `yolov8s.pt`, `yolov8m.pt`, etc. for better accuracy |
| Confidence | `conf >= 0.3` | Minimum detection confidence |

### Creating your own mask

Make a black image the same size as your video, then paint the road area white
(any image editor works). Black regions are ignored by the detector.

## Known Issues

- **Condition precedence:** in the class check, `and conf>=0.3` only applies to
  `truck` because `and` binds tighter than `or`. Use:
  ```python
  if class_name in ("car", "motorcycle", "bus", "truck") and conf >= 0.3:
  ```
- **Missing parentheses:** use `cv2.destroyAllWindows()` at the end of the script.
- **End of video:** add `if not success: break` right after `cap.read()`, otherwise
  the script crashes when the video ends.
- **Mask size:** if the video isn't 1280x720, `cv2.bitwise_and` will fail. Resize the
  mask: `mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))`.
- **Unused variable:** `limits` is defined but not used anywhere.

## Requirements

- Python 3.8+
- See `requirements.txt` for packages

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [cvzone](https://github.com/cvzone/cvzone)
