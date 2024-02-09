
from ultralytics import YOLO

model = YOLO("yolov8m.pt")

results = model.train(data="/content/drive/MyDrive/Plants Diseases Detection and Classification.v12i.yolov8/data.yaml",epochs=50,lr0=0.0005,augment=True)