from ultralytics import YOLO

#проверки обученной модели на тестовых изображениях.
def main():
    model = YOLO("runs/detect/runs/drone_yolov8n/weights/best.pt")
    results = model.predict(
        source="dataset/images/test",
        conf=0.25,
        save=True
    )

    print("Распознавание завершено.")
    print(f"Обработано изображений: {len(results)}")


if __name__ == "__main__":
    main()