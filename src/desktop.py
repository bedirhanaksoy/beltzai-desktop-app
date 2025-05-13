import tkinter as tk
from tkinter import messagebox
import requests
import cv2
from PIL import Image, ImageTk
from pathlib import Path
from compare_and_track import Comparer
from detection_and_comparison import DetectionAndComparison
import time

SERVER_URL = "http://example.com/validate_session"  # Replace with your actual endpoint

MAIN_PATH = Path(__file__).resolve()
resources_path = MAIN_PATH.resolve().parent.parent / "resources"

models_path = str(resources_path / "models/right_part_medium.pt")
right_base_image_path = str(resources_path / "base_images/right_base_image.png")
left_base_image_path = str(resources_path / "base_images/left_base_image.png")

boxes = [
            [(50, 200), (230, 380)],  # Right box
            [(350, 200), (530, 380)]   # Left box
        ]


def crop_and_save(box, frame, filename):
    """Save cropped region as base image with an additional 30 pixels margin"""
    if box and len(box) == 2:
        x1, y1 = box[0]
        x2, y2 = box[1]
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Add 80 pixels margin
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
        self.update_frame_id = None  # Add this to track the update_frame loop
        self._build_login_screen()

    def _build_login_screen(self):

        # If we have a cap from previous screen, release it
        if self.cap:
            self.cap.release()
            self.cap = None

        # Cancel any pending update
        if self.update_frame_id:
            self.after_cancel(self.update_frame_id)
            self.update_frame_id = None
            
        # Clear any existing widgets
        for widget in self.winfo_children():
            widget.destroy()
        if self.cap:
            self.cap.release()
            self.cap = None

        # Session ID entry
        tk.Label(self, text="Enter Session ID:").pack(pady=10)
        self.session_entry = tk.Entry(self)
        self.session_entry.pack(pady=5)

        # Submit button
        tk.Button(self, text="Submit", command=self._validate_session).pack(pady=20)

    def _validate_session(self):
        #self._build_operation_screen() # For testing purposes, directly go to the base image screen
        self._build_base_image_screen()
        return

        session_id = self.session_entry.get().strip()
        if not session_id:
            messagebox.showwarning("Input Error", "Please enter a session ID.")
            return

        try:
            response = requests.post(SERVER_URL, json={"session_id": session_id})
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            messagebox.showerror("Connection Error", f"Failed to connect to server: {e}")
            return
        except ValueError:
            messagebox.showerror("Response Error", "Invalid response from server.")
            return

        if data.get("valid"):
            self._build_base_image_screen()
        else:
            messagebox.showerror("Invalid Session", "The session ID is not valid. Please try again.")

    def _build_base_image_screen(self):
        # Cancel any pending update
        if self.update_frame_id:
            self.after_cancel(self.update_frame_id)
            self.update_frame_id = None
            
        # Clear widgets
        for widget in self.winfo_children():
            widget.destroy()
        if self.cap:
            self.cap.release()
            
        # Initialize camera
        self.cap = cv2.VideoCapture(0)

        # Canvas for video
        self.video_label = tk.Label(self)
        self.video_label.place(x=0, y=0, width=800, height=600)

        # Instruction label
        self.instruction = tk.Label(self, text="Place objects on the boxes and press 't' to take base images",
                                    bg='black', fg='white', font=(None, 12))
        self.instruction.place(x=200, y=10)

        # Next button disabled until at least one capture
        self.next_button = tk.Button(self, text="Next", state=tk.DISABLED, command=self._build_operation_screen)
        self.next_button.place(x=700, y=500)

        # End session button
        self.end_button = tk.Button(self, text="End Session", command=self._end_session)
        self.end_button.place(x=10, y=550)

        # Bind keypress
        #self.bind('<t>', self._capture_base_image)
        self.bind('<t>', self._skip_capturing_base_image)

        # Start video loop
        self._update_frame()

    def _skip_capturing_base_image(self, event=None):
        self.next_button.config(state=tk.NORMAL)

    def _update_frame(self):
        if not self.cap or not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if ret:
            # Flip and resize
            #frame = cv2.flip(frame, 1)
            #frame = cv2.resize(frame, (800, 600))
            
            # Draw two green rectangles where objects should be placed
            for i, box in enumerate(boxes):
                (x1, y1), (x2, y2) = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Add text for each box
                label = "Right Box" if i == 0 else "Left Box"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Convert to ImageTk
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
            
        # Schedule the next update and save the ID
        self.update_frame_id = self.after(10, self._update_frame)

    def _capture_base_image(self, event=None):
        # Capture current frame
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
        
        # Enable the Next button after capturing base images
        self.next_button.config(state=tk.NORMAL)

    def _build_operation_screen(self):
        # Cancel any pending update
        if self.update_frame_id:
            self.after_cancel(self.update_frame_id)
            self.update_frame_id = None

        # Clear existing widgets
        for widget in self.winfo_children():
            widget.destroy()

        # If we have a cap from the previous screen, release it
        if self.cap:
            self.cap.release()
            self.cap = None

        # Create a frame for the operation screen
        operation_frame = tk.Frame(self, width=800, height=600)
        operation_frame.pack(fill="both", expand=True)

        # Instantiate and run DetectionAndComparison
        self.detection_and_comparison = DetectionAndComparison(
            tkinter_frame=operation_frame,
            end_session_callback=self._end_session,
            model_path=models_path,
            right_base_image_path=right_base_image_path,
            left_base_image_path=left_base_image_path
        )
        self.detection_and_comparison.run()

    def _end_session(self):
        # Stop DetectionAndComparison if running
        if hasattr(self, 'detection_and_comparison') and self.detection_and_comparison.is_running:
            self.detection_and_comparison._stop_process()

        # Go back to the login screen
        self._build_login_screen()

    def on_closing(self):
        # Cancel any pending update
        if self.update_frame_id:
            self.after_cancel(self.update_frame_id)
            
        if self.cap:
            self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = SequenceApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
