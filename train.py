from ultralytics import YOLO

#обучение модели
def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="dataset/dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device="cpu",
        project="runs",
        name="drone_yolov8n",
        plots=True
    )


if __name__ == "__main__":
    main()