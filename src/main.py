from compare_and_track import Comparer
import cv2
import numpy as np
from collections import deque
import time
from ultralytics import YOLO
from pathlib import Path

MAIN_PATH = Path(__file__).resolve()
resources_path = MAIN_PATH.resolve().parent.parent / "resources"

models_path = str(resources_path / "models/right_part_medium.pt")
right_base_image_path = str(resources_path / "base_images/right_base_image.png")
left_base_image_path = str(resources_path / "base_images/left_base_image.png")


if __name__ == "__main__":
    # create a comparer object
    comparer = Comparer(camera_id=0, model_path=models_path)
    comparer.load_base_images()

    """Main loop for camera feed and interaction"""
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
            
            # x1, y1, x2, y2, conf, cls = det[:6]
            
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
