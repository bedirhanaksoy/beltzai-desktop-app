from compare_and_track import Comparer
import cv2
import numpy as np
from collections import deque
import time
from ultralytics import YOLO
from pathlib import Path
import tkinter as tk
from PIL import Image, ImageTk

MAIN_PATH = Path(__file__).resolve()
resources_path = MAIN_PATH.resolve().parent.parent / "resources"

models_path = str(resources_path / "models/right_part_medium.pt")
right_base_image_path = str(resources_path / "base_images/right_base_image.png")
left_base_image_path = str(resources_path / "base_images/left_base_image.png")


def detect_and_compare_run(tkinter_frame=None, end_session_callback=None):
    # create a comparer object
    comparer = Comparer(camera_id=2, model_path=models_path)
    comparer.load_base_images()

    # For standalone cv2 window (original functionality)
    if tkinter_frame is None:
        cv2.namedWindow('Camera Feed')

        while True:
            ret, comparer.frame = comparer.cap.read()
            if not ret:
                break

            comparer.frame_display = comparer.frame.copy()
            comparer.print_boxes()

            # Run YOLO detection
            comparer.yolo_detections = comparer.model.track(comparer.frame, verbose=False, persist=True)
            current_time = time.time()
            
            # Check if any object is inside the box
            comparer.is_right_box_empty, comparer.is_left_box_empty = comparer.check_if_box_is_empty(comparer.yolo_detections[0].boxes.data) 
            # Process each detection

            for det in comparer.yolo_detections[0].boxes:
                
                track_id = det.id.int().cpu().numpy()[0] if det.id is not None else 0
                # Extract other detection data
                x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
                conf = det.conf.cpu().numpy()[0]
                cls = det.cls.cpu().numpy()[0]

                if conf < 0.5:  # Confidence threshold
                    continue
                # Check for each test box
                comparer.compare(x1, y1, x2, y2, cls, track_id, current_time)
                comparer.check(x1, x2, track_id)
                # Draw detection
                cv2.rectangle(comparer.frame_display, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(comparer.frame_display, str(comparer.index_side_info[track_id]), (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Display frame
            cv2.imshow('Camera Feed', comparer.frame_display)

            # Handle key events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1') and len(comparer.boxes) >= 1:
                comparer.crop_and_save(comparer.boxes[0], right_base_image_path)
            elif key == ord('2') and len(comparer.boxes) >= 2:
                comparer.crop_and_save(comparer.boxes[1], left_base_image_path)
            elif key == ord('c'):
                comparer.boxes = []

        comparer.logger.save_session()
        comparer.cap.release()
        cv2.destroyAllWindows()
        
    # For Tkinter integration
    else:
        # Create necessary Tkinter elements for displaying the video feed
        video_label = tk.Label(tkinter_frame)
        video_label.place(x=0, y=0, width=800, height=450)
        
        # Status display
        status_label = tk.Label(tkinter_frame, text="Processing...", font=("Arial", 12))
        status_label.place(x=10, y=460)
        
        # Add buttons for the functionality that was previously bound to key presses
        
        stop_button = tk.Button(
            tkinter_frame, 
            text="End Session", 
            command=lambda: stop_process()
        )
        stop_button.place(x=650, y=500)
        
        is_running = True
        
        def stop_process():
            nonlocal is_running
            is_running = False
            # Add cleanup operations here to consolidate the logic
            comparer.logger.save_session()
            comparer.cap.release()
            if end_session_callback:
                end_session_callback()
        
        def update_frame():
            if not is_running:
                return
                
            ret, comparer.frame = comparer.cap.read()
            if not ret:
                stop_process()  # This will handle all cleanup and callbacks
                return

            comparer.frame_display = comparer.frame.copy()
            comparer.print_boxes()

            # Run YOLO detection
            comparer.yolo_detections = comparer.model.track(comparer.frame, verbose=False, persist=True)
            current_time = time.time()
            
            # Check if any object is inside the box
            comparer.is_right_box_empty, comparer.is_left_box_empty = comparer.check_if_box_is_empty(comparer.yolo_detections[0].boxes.data) 
            
            # Update status in Tkinter interface
            right_status = "Empty" if comparer.is_right_box_empty else "Occupied"
            left_status = "Empty" if comparer.is_left_box_empty else "Occupied"
            status_label.config(text=f"Right Box: {right_status} | Left Box: {left_status}")
            
            # Process each detection
            for det in comparer.yolo_detections[0].boxes:
                track_id = det.id.int().cpu().numpy()[0] if det.id is not None else 0
                x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
                conf = det.conf.cpu().numpy()[0]
                cls = det.cls.cpu().numpy()[0]

                if conf < 0.5:  # Confidence threshold
                    continue
                    
                # Check for each test box
                comparer.compare(x1, y1, x2, y2, cls, track_id, current_time)
                comparer.check(x1, x2, track_id)
                
                # Draw detection on frame_display (will be converted to Tkinter format)
                cv2.rectangle(comparer.frame_display, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(comparer.frame_display, str(comparer.index_side_info[track_id]), 
                           (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Convert to Tkinter format and display
            img_rgb = cv2.cvtColor(comparer.frame_display, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            video_label.imgtk = img_tk
            video_label.configure(image=img_tk)
            
            # Continue the loop
            tkinter_frame.after(10, update_frame)
        
        # Start the update loop
        update_frame()


if __name__ == "__main__":
    detect_and_compare_run()
