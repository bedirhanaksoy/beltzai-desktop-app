import cv2
import time
from PIL import Image, ImageTk
import tkinter as tk
from comparer_module import Comparer
from sticker_module import detect_stickers  # Returns only left stickers now

class SessionOperator:
    def __init__(self, tkinter_frame, end_session_callback, model_path, right_base_image_path, left_base_image_path):
        self.tkinter_frame = tkinter_frame
        self.tkinter_frame.winfo_toplevel().geometry("1000x800")
        self.end_session_callback = end_session_callback
        self.model_path = model_path
        self.right_base_image_path = right_base_image_path
        self.left_base_image_path = left_base_image_path
        self.comparer = Comparer(camera_id=2, model_path=self.model_path)
        self.comparer.load_base_images()
        self.is_running = True
   
    def run(self):
        # Set background color of entire frame to match previous pages
        self.tkinter_frame.config(bg="#E9EBFF")

        # Top line
        top_line = tk.Frame(self.tkinter_frame, bg="#374151", height=3)
        top_line.pack(fill="x", pady=(0, 10))

        # Video display area
        self.video_label = tk.Label(self.tkinter_frame, bg="#E9EBFF")
        self.video_label.pack(fill="both", expand=True)

        # Overlay for pause message
        self.pause_overlay = tk.Label(
            self.video_label,
            text="Kontrol Durduruldu",
            font=("Arial", 24, "bold"),
            fg="white",
            bg="#A8A8A8"
        )
        self.pause_overlay.pack_forget()

        # Status label
        self.status_label = tk.Label(self.tkinter_frame, text="Processing...", font=("Arial", 12), bg="#E9EBFF")
        self.status_label.pack(pady=5)

        # Button frame
        button_frame = tk.Frame(self.tkinter_frame, bg="#E9EBFF")
        button_frame.pack(pady=10)

        self.stop_button = tk.Button(
            button_frame,
            text="Durdur",
            bg="#4856D4",
            fg="white",
            font=("Arial", 11, "bold"),
            relief="raised",
            bd=2,
            padx=30,
            pady=5,
            command=self._stop_updates
        )
        self.stop_button.pack(side="left", padx=10)

        self.continue_button = tk.Button(
            button_frame,
            text="Devam Et",
            bg="#4CD964",
            fg="white",
            font=("Arial", 11, "bold"),
            relief="raised",
            bd=2,
            padx=30,
            pady=5,
            command=self._continue_updates
        )
        self.continue_button.pack_forget()

        self.end_button = tk.Button(
            button_frame,
            text="Oturumu Bitir",
            bg="#FF0000",
            fg="white",
            font=("Arial", 11, "bold"),
            relief="raised",
            bd=2,
            padx=30,
            pady=5,
            command=self._stop_process
        )
        self.end_button.pack(side="right", padx=10)

        # Bottom line
        bottom_line = tk.Frame(self.tkinter_frame, bg="#374151", height=3)
        bottom_line.pack(fill="x", side="bottom", pady=(10, 0))

        self._update_frame()

    def _stop_updates(self):
        """Pause frame updates without ending the session."""
        self.is_running = False
        self.stop_button.pack_forget()
        self.continue_button.pack(side="left", padx=10)

        # Start blinking
        self._blink_pause_overlay()
    
    def _continue_updates(self):
        """Resume frame updates."""
        self.is_running = True
        self.continue_button.pack_forget()
        self.stop_button.pack(side="left", padx=10)

        # Ensure pause text is hidden
        self.pause_overlay.pack_forget()

        self._update_frame()
    
    def _blink_pause_overlay(self):
        if not self.is_running:
            # Toggle visibility
            if self.pause_overlay.winfo_ismapped():
                self.pause_overlay.pack_forget()
            else:
                self.pause_overlay.pack(anchor="n", pady=10)

            # Schedule next blink
            self.tkinter_frame.after(500, self._blink_pause_overlay)

    def _update_frame(self):
        if not self.is_running:
            return

        ret, self.comparer.frame = self.comparer.cap.read()
        if not ret:
            self._stop_process()
            return

        self.comparer.frame_display = self.comparer.frame.copy()
        self.comparer.print_boxes()

        self.comparer.yolo_detections = self.comparer.model.track(self.comparer.frame, verbose=False, persist=True)
        current_time = time.time()

        self.comparer.is_right_box_empty, self.comparer.is_left_box_empty = self.comparer.check_if_box_is_empty(
            self.comparer.yolo_detections[0].boxes.data
        )

        right_status = "Empty" if self.comparer.is_right_box_empty else "Occupied"
        left_status = "Empty" if self.comparer.is_left_box_empty else "Occupied"
        self.status_label.config(text=f"Right Box: {right_status} | Left Box: {left_status}")

        # Detect both left and right stickers per frame
        all_left_stickers, all_right_stickers = detect_stickers(self.comparer.frame, conf_threshold=0.7)

        for det in self.comparer.yolo_detections[0].boxes:
            track_id = det.id.int().cpu().numpy()[0] if det.id is not None else 0
            x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
            conf = det.conf.cpu().numpy()[0]
            cls = det.cls.cpu().numpy()[0]

            if conf < 0.5:
                continue

            part_side = self.comparer.index_side_info[track_id]  # 1 = right, 2 = left

            self.comparer.compare(x1, y1, x2, y2, cls, track_id, current_time)
            self.comparer.check(x1, x2, track_id)

            # Check for left stickers inside this part
            for box in all_left_stickers:
                sx1, sy1, sx2, sy2 = box.xyxy[0].tolist()
                cx, cy = (sx1 + sx2) / 2, (sy1 + sy2) / 2
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    cv2.rectangle(self.comparer.frame_display, (int(sx1), int(sy1)), (int(sx2), int(sy2)), (0, 255, 0), 2)
                    cv2.putText(self.comparer.frame_display, "L", (int(sx1), int(sy1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    if part_side == 1:  # wrong: left sticker on right-labeled part
                        self.comparer.sticker_warning_timestamp = time.time()
                        self.comparer.sticker_error_type = "left_on_right"

            # Check for right stickers inside this part
            for box in all_right_stickers:
                sx1, sy1, sx2, sy2 = box.xyxy[0].tolist()
                cx, cy = (sx1 + sx2) / 2, (sy1 + sy2) / 2
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    cv2.rectangle(self.comparer.frame_display, (int(sx1), int(sy1)), (int(sx2), int(sy2)), (0, 0, 255), 2)
                    cv2.putText(self.comparer.frame_display, "R", (int(sx1), int(sy1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    if part_side == 2:  # wrong: right sticker on left-labeled part
                        self.comparer.sticker_warning_timestamp = time.time()
                        self.comparer.sticker_error_type = "right_on_left"

            # Draw part bounding box
            cv2.rectangle(self.comparer.frame_display, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(self.comparer.frame_display, str(part_side),
                        (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show warning messages if triggered in the last 1 second
        if hasattr(self.comparer, 'sticker_warning_timestamp') and time.time() - self.comparer.sticker_warning_timestamp < 1:
            if hasattr(self.comparer, 'sticker_error_type'):
                if self.comparer.sticker_error_type == "left_on_right":
                    cv2.putText(self.comparer.frame_display, "ERROR: Left Sticker on Right Part!",
                                (self.comparer.frame_display.shape[1] // 2 - 250, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
                elif self.comparer.sticker_error_type == "right_on_left":
                    cv2.putText(self.comparer.frame_display, "ERROR: Right Sticker on Left Part!",
                                (self.comparer.frame_display.shape[1] // 2 - 250, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

        # Show frame on GUI
        img_rgb = cv2.cvtColor(self.comparer.frame_display, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.imgtk = img_tk
        self.video_label.configure(image=img_tk)

        self.tkinter_frame.after(2, self._update_frame)

    def _stop_process(self):
        self.is_running = False
        self.comparer.logger.save_session()
        self.comparer.cap.release()
        if self.end_session_callback:
            self.end_session_callback()
