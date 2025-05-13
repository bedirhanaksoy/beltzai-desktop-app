import tkinter as tk
from tkinter import messagebox
import requests
import cv2
from PIL import Image, ImageTk
from pathlib import Path
from compare_and_track import Comparer
from detection_and_comparison import ConveyorBeltOperations
import time

SERVER_URL = "http://example.com/validate_session"  # Replace with your actual endpoint

MAIN_PATH = Path(__file__).resolve()
resources_path = MAIN_PATH.resolve().parent.parent / "resources"

models_path = str(resources_path / "models/right_part_medium.pt")
right_base_image_path = str(resources_path / "base_images/right_base_image.png")
left_base_image_path = str(resources_path / "base_images/left_base_image.png")

boxes = [
    [(50, 200), (230, 380)],  # Right box
    [(350, 200), (530, 380)]  # Left box
]


def crop_and_save(box, frame, filename):
    """Save cropped region as base image with an additional margin"""
    if box and len(box) == 2:
        x1, y1 = box[0]
        x2, y2 = box[1]
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        x1 = max(0, x1 - 65)
        y1 = max(0, y1 - 65)
        x2 = min(frame.shape[1], x2 + 65)
        y2 = min(frame.shape[0], y2 + 65)

        cropped = frame[y1:y2, x1:x2]
        cv2.imwrite(filename, cropped)
        print(f"Saved {filename}")


class SequenceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sequence App")
        self.geometry("800x600")
        self.resizable(False, False)
        self.cap = None
        self.update_frame_id = None
        self._build_login_screen()

    def _build_login_screen(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.update_frame_id:
            self.after_cancel(self.update_frame_id)
            self.update_frame_id = None

        for widget in self.winfo_children():
            widget.destroy()

        login_frame = tk.Frame(self, bg="white", width=800, height=600)
        login_frame.pack(fill="both", expand=True)

        center_frame = tk.Frame(login_frame, bg="white")
        center_frame.place(relx=0.5, rely=0.5, anchor="center")

        title_label = tk.Label(center_frame, text="Enter Session ID:", bg="white", font=("Arial", 18))
        title_label.pack(pady=12)

        self.session_entry = tk.Entry(center_frame, font=("Arial", 16), width=32)
        self.session_entry.pack(pady=12, ipady=6)

        submit_button = tk.Button(
            center_frame,
            text="Submit",
            command=self._validate_session,
            bg="#007BFF",
            fg="white",
            font=("Arial", 14, "bold"),
            relief="raised",
            bd=4,
            padx=24,
            pady=12
        )
        submit_button.pack(pady=24)

    def _validate_session(self):
        self._build_base_image_screen()
        return

    def _build_base_image_screen(self):
        if self.update_frame_id:
            self.after_cancel(self.update_frame_id)
            self.update_frame_id = None

        for widget in self.winfo_children():
            widget.destroy()
        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(0)

        self.video_label = tk.Label(self)
        self.video_label.place(x=0, y=0, width=800, height=600)

        self.instruction = tk.Label(
            self,
            text="Place objects on the boxes and press 't' to take base images",
            bg='black',
            fg='white',
            font=("Arial", 14)
        )
        self.instruction.place(x=250, y=10)

        self.next_button = tk.Button(
            self,
            text="Next",
            state=tk.DISABLED,
            command=self._build_operation_screen,
            bg="#28a745",
            fg="white",
            font=("Arial", 12, "bold"),
            relief="raised",
            bd=4,
            padx=20,
            pady=10
        )
        self.next_button.place(x=700, y=500)

        self.end_button = tk.Button(
            self,
            text="End Session",
            command=self._end_session,
            bg="#dc3545",
            fg="white",
            font=("Arial", 12, "bold"),
            relief="raised",
            bd=4,
            padx=20,
            pady=10
        )
        self.end_button.place(x=10, y=550)

        self.bind('<t>', self._skip_capturing_base_image)
        self._update_frame()

    def _skip_capturing_base_image(self, event=None):
        self.next_button.config(state=tk.NORMAL)

    def _update_frame(self):
        if not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if ret:
            for i, box in enumerate(boxes):
                (x1, y1), (x2, y2) = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = "Right Box" if i == 0 else "Left Box"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

        self.update_frame_id = self.after(10, self._update_frame)

    def _capture_base_image(self, event=None):
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Capture Error", "Camera not available.")
            return

        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Capture Error", "Failed to capture image.")
            return

        if len(boxes) == 2:
            crop_and_save(boxes[0], frame, right_base_image_path)
            crop_and_save(boxes[1], frame, left_base_image_path)

        self.next_button.config(state=tk.NORMAL)

    def _build_operation_screen(self):
        if self.update_frame_id:
            self.after_cancel(self.update_frame_id)
            self.update_frame_id = None

        for widget in self.winfo_children():
            widget.destroy()

        if self.cap:
            self.cap.release()
            self.cap = None

        operation_frame = tk.Frame(self, width=800, height=600)
        operation_frame.pack(fill="both", expand=True)

        self.detection_and_comparison = ConveyorBeltOperations(
            tkinter_frame=operation_frame,
            end_session_callback=self._end_session,
            model_path=models_path,
            right_base_image_path=right_base_image_path,
            left_base_image_path=left_base_image_path
        )
        self.detection_and_comparison.run()

    def _end_session(self):
        if hasattr(self, 'detection_and_comparison') and self.detection_and_comparison.is_running:
            self.detection_and_comparison._stop_process()
        self._build_login_screen()

    def on_closing(self):
        if self.update_frame_id:
            self.after_cancel(self.update_frame_id)
        if self.cap:
            self.cap.release()
        self.destroy()


if __name__ == "__main__":
    app = SequenceApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
