import cv2
import time
from PIL import Image, ImageTk
import tkinter as tk
from compare_and_track import Comparer
from sticker_module import detect_stickers  # Returns only left stickers now

class ConveyorBeltOperations:
    def __init__(self, tkinter_frame, end_session_callback, model_path, right_base_image_path, left_base_image_path):
        self.tkinter_frame = tkinter_frame
        self.end_session_callback = end_session_callback
        self.model_path = model_path
        self.right_base_image_path = right_base_image_path
        self.left_base_image_path = left_base_image_path
        self.comparer = Comparer(camera_id=2, model_path=self.model_path)
        self.comparer.load_base_images()
        self.is_running = True

    def run(self):
        self.video_label = tk.Label(self.tkinter_frame)
        self.video_label.place(x=0, y=0, width=800, height=450)

        self.status_label = tk.Label(self.tkinter_frame, text="Processing...", font=("Arial", 12))
        self.status_label.place(x=10, y=460)

        stop_button = tk.Button(
            self.tkinter_frame,
            text="End Session",
            command=self._stop_process
        )
        stop_button.place(x=650, y=500)

        self._update_frame()
        
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

        # Detect only left stickers once per frame
        all_left_stickers = detect_stickers(self.comparer.frame)

        for det in self.comparer.yolo_detections[0].boxes:
            track_id = det.id.int().cpu().numpy()[0] if det.id is not None else 0
            x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
            conf = det.conf.cpu().numpy()[0]
            cls = det.cls.cpu().numpy()[0]

            if conf < 0.5:
                continue

            part_box = [x1, y1, x2, y2]
            part_side = self.comparer.index_side_info[track_id]  # 1 = right, 2 = left

            # Call compare and check synchronously
            self.comparer.compare(x1, y1, x2, y2, cls, track_id, current_time)
            self.comparer.check(x1, x2, track_id)

            # Draw stickers and check placement correctness
            for box in all_left_stickers:
                sx1, sy1, sx2, sy2 = box.xyxy[0].tolist()
                sticker_center_x = (sx1 + sx2) / 2
                sticker_center_y = (sy1 + sy2) / 2

                # Check if sticker center lies inside part box
                if x1 <= sticker_center_x <= x2 and y1 <= sticker_center_y <= y2:
                    # Draw the sticker
                    x1s, y1s, x2s, y2s = map(int, [sx1, sy1, sx2, sy2])
                    cv2.rectangle(self.comparer.frame_display, (x1s, y1s), (x2s, y2s), (0, 255, 0), 2)
                    cv2.putText(self.comparer.frame_display, "L", (x1s, y1s - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    # If it's a left sticker on a right-labeled part → trigger warning
                    if part_side == 1:
                        self.comparer.sticker_warning_timestamp = time.time()

            # Draw part box
            cv2.rectangle(self.comparer.frame_display, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(self.comparer.frame_display, str(part_side),
                        (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show warning if triggered in last 1 second
        if time.time() - self.comparer.sticker_warning_timestamp < 1:
            cv2.putText(self.comparer.frame_display, "ERROR: Left Sticker on Right Part!",
                        (self.comparer.width // 2 - 250, 60),
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
