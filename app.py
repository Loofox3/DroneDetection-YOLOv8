import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2


MODEL_PATH = "runs/detect/runs/drone_yolov8n/weights/best.pt"


class DroneDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Распознавание БПЛА")
        self.root.geometry("1000x700")

        self.model = YOLO(MODEL_PATH)

        self.title_label = tk.Label(
            root,
            text="Система обнаружения и классификации БПЛА",
            font=("Arial", 18, "bold")
        )
        self.title_label.pack(pady=10)

        self.open_button = tk.Button(
            root,
            text="Загрузить изображение",
            font=("Arial", 14),
            command=self.open_image
        )
        self.open_button.pack(pady=10)

        self.result_label = tk.Label(
            root,
            text="Загрузите изображение для распознавания",
            font=("Arial", 12)
        )
        self.result_label.pack(pady=10)

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

    def open_image(self):
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[
                ("Изображения", "*.jpg *.jpeg *.png"),
                ("Все файлы", "*.*")
            ]
        )

        if not file_path:
            return

        try:
            results = self.model.predict(
                source=file_path,
                conf=0.25,
                save=False
            )

            result = results[0]
            result_image = result.plot()

            result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(result_image)

            pil_image.thumbnail((900, 500))

            tk_image = ImageTk.PhotoImage(pil_image)

            self.image_label.config(image=tk_image)
            self.image_label.image = tk_image

            boxes = result.boxes

            if len(boxes) == 0:
                self.result_label.config(
                    text="БПЛА на изображении не обнаружен."
                )
            else:
                output_text = f"Обнаружено объектов: {len(boxes)}\n"

                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.model.names[class_id]

                    output_text += (
                        f"Класс: {class_name} | "
                        f"Вероятность: {confidence:.2%}\n"
                    )

                self.result_label.config(text=output_text)

        except Exception as error:
            messagebox.showerror(
                "Ошибка",
                f"Не удалось обработать изображение:\n{error}"
            )


if __name__ == "__main__":
    root = tk.Tk()
    app = DroneDetectionApp(root)
    root.mainloop()