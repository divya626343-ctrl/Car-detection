from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture("2.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
mask = cv2.imread("mask.png")

limits = [500, 290, 900, 290]


model = YOLO('yolov8n.pt')

while True:
    success, frame = cap.read()
    imRegion = cv2.bitwise_and(frame, mask)
    results = model(imRegion, stream = True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
          x1, y1, x2, y2 = box.xyxy[0]
          x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
        #   cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

          w, h = x2-x1, y2-y1
    
          cvzone.cornerRect(frame, bbox=(x1, y1, w, h))
          conf = math.ceil((box.conf[0]*100))/100
          cls = int(box.cls[0])
          class_name = model.names[cls]
          if class_name == "car" or class_name == "motorcycle" or class_name == "bus" or class_name == "truck" and conf>=0.3:
             cvzone.putTextRect(frame, f"{class_name, conf}", (max(0, x1), max(32, y1)), scale = 0.6, thickness=1, offset = 3)


    cv2.imshow("image", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows
