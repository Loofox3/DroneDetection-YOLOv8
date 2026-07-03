from ultralytics import YOLO

#Нужен для оценки качества модели. После запуска YOLO создаст метрики, матрицу ошибок и графики.
def main():
    model = YOLO("runs/detect/runs/drone_yolov8n/weights/best.pt")

    metrics = model.val(
        data="dataset/dataset.yaml",
        split="test",
        plots=True
    )

    print("Оценка модели завершена.")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()