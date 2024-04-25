from ultralytics import YOLO

def main():
    #model = YOLO(r'C:\Projects\image-detection\runs\detect\train8\weights\best.pt') 
    model = YOLO("yolov8n.pt")
    model.train(data='C:\Projects\image-detection\osrs_dataset\data.yaml', epochs=100, imgsz=640, device=0)

if __name__ == "__main__":
    main()