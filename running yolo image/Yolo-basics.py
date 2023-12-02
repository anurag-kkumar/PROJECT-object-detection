from ultralytics import YOLO
import cv2


model= YOLO('../YOLO-weight/yolov8n.pt') # n for low frequncy of object detection
results= model("image/3.png", show=True)
cv2.waitKey(0)