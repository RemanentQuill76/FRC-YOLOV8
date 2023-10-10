from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor
import cv2

model = YOLO(r"C:\Users\Owner\Desktop\robot shit\v17_1536x864.pt")

results = model.predict(source="0", show=True)

print(results)