import tkinter as tk
from tkinter import messagebox
import requests
import cv2
from PIL import Image, ImageTk
from main import detect_and_compare_run  # Make sure this import works as expected

SERVER_URL = "http://example.com/validate_session"  # Replace with your actual endpoint

class SequenceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sequence App")
        self.geometry("800x600")
        self.resizable(False, False)
        self.cap = None
        self.base_images = []
        self._build_login_screen()

    def _build_login_screen(self):
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
        # Clear widgets
        for widget in self.winfo_children():
            widget.destroy()
        # Initialize camera
        self.cap = cv2.VideoCapture(2)

        # Canvas for video
        self.video_label = tk.Label(self)
        self.video_label.place(x=0, y=0, width=800, height=600)

        # Instruction label
        self.instruction = tk.Label(self, text="Place objects on the boxes and press 't' to take base images",
                                    bg='black', fg='white', font=(None, 12))
        self.instruction.place(x=200, y=10)

        # Next button disabled until at least one capture
        self.next_button = tk.Button(self, text="Next", state=tk.DISABLED, command=self._on_next)
        self.next_button.place(x=700, y=500)

        # End session button
        self.end_button = tk.Button(self, text="End Session", command=self._end_session)
        self.end_button.place(x=10, y=550)

        # Bind keypress
        self.bind('<t>', self._capture_base_image)

        # Start video loop
        self._update_frame()

    def _update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Flip and resize
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (800, 600))
            # Draw two green rectangles where objects should be placed
            h, w, _ = frame.shape
            box_w, box_h = 200, 200
            # Left box
            cv2.rectangle(frame, (int(w*0.2 - box_w/2), int(h/2 - box_h/2)),
                          (int(w*0.2 + box_w/2), int(h/2 + box_h/2)), (0, 255, 0), 2)
            # Right box
            cv2.rectangle(frame, (int(w*0.8 - box_w/2), int(h/2 - box_h/2)),
                          (int(w*0.8 + box_w/2), int(h/2 + box_h/2)), (0, 255, 0), 2)

            # Convert to ImageTk
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
        # Loop
        self.after(10, self._update_frame)

    def _capture_base_image(self, event=None):
        # Capture current frame
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Capture Error", "Failed to capture image.")
            return
        # Save or store
        self.base_images.append(frame.copy())
        messagebox.showinfo("Captured", f"Base image #{len(self.base_images)} captured successfully.")
        # Enable Next button
        if len(self.base_images) >= 1:
            self.next_button.config(state=tk.NORMAL)

    def _on_next(self):
        # Call detect_and_compare function here
        try:
            detect_and_compare_run()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during detection and comparison: {e}")

    def _end_session(self):
        # Go back to the login screen
        self.base_images.clear()
        self._build_login_screen()

    def on_closing(self):
        if self.cap:
            self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = SequenceApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
