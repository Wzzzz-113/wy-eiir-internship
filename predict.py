import os
from ultralytics import YOLO

model = YOLO("/home/wy/YOLO/yolo_train/runs/detect/train2/weights/best.pt")

input_folder = "/home/wy/YOLO/yolo_train/decoder_0818_test/"
suffix = "png"
for filename in os.listdir(input_folder):
    if filename.endswith(suffix):
        image_path = os.path.join(input_folder, filename)
        print(f"image path : {image_path}")

        results = model.predict(source=image_path, conf=0.8, iou=0.1, save=True, save_txt=True)
        print(f"image {filename} predict done")
        
