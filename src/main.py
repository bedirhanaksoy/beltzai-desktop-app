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
        
        # Draw boxes and labels
        for i, box in enumerate(comparer.boxes):
            # Get box coordinates
            x1, y1 = box[0]
            x2, y2 = box[1]
            
            # Draw box
            color = (0, 255, 0) if (i == 0 and comparer.right_box_color == 0) or (i == 1 and comparer.left_box_color == 0) else (0, 0, 255)
            cv2.rectangle(comparer.frame_display, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Add label
            label = "Right Box" if i == 0 else "Left Box"
            # Position the label above the box
            label_x = x1
            label_y = y1 - 10
            cv2.putText(comparer.frame_display, label, 
                        (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
            
            # Calculate and display area
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            area = width * height
            area_text = f"Area: {area}px"
            cv2.putText(comparer.frame_display, area_text,
                        (label_x, label_y - 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
        
        # Run YOLO detection
        comparer.yolo_detections = comparer.model(comparer.frame, verbose=False)
        current_time = time.time()
        
        # Check if any object is inside the box
        comparer.is_right_box_empty, comparer.is_left_box_empty = comparer.check_if_box_is_empty(comparer.yolo_detections[0].boxes.data) 

        # Process each detection
        for det in comparer.yolo_detections[0].boxes.data:
            x1, y1, x2, y2, conf, cls = det 
            if conf < 0.5:  # Confidence threshold
                continue
            # Check for each test box
            for box_idx, box in enumerate(comparer.boxes):
                #print("left box state: ", comparer.left_box_state)
                #print("right box state: ", comparer.right_box_state)
                box_data = comparer.objects_in_boxes[box_idx]
                if (box_idx == 0 and comparer.right_box_state == 0) or (box_idx == 1 and comparer.left_box_state == 0):
                    if comparer.check_object_in_box(box, (x1, y1, x2, y2)):
                        # New object detected in box
                        if box_data['object'] is None:
                            box_data['object'] = cls.item()
                            box_data['start_time'] = current_time
                            box_data['test_results'] = []
                            box_data['bbox_history'].clear()
                            
                        # Add current bbox to history
                        current_bbox = (x1, y1, x2, y2)
                        box_data['bbox_history'].append(current_bbox)
                        
                        # Check if object has been still for threshold time
                        if (current_time - box_data['start_time'] >= comparer.STILL_THRESHOLD):
                            if comparer.check_if_object_stable(current_bbox, box_data['bbox_history']):
                                # Run template matching test
                                results = comparer.test_frame(comparer.frame, box_idx=box_idx)
                                if results:
                                    #print("results size: ", len(results))
                                    #print("results[box_idx]['warning']: ", results[box_idx]['warning'])
                                    box_data['test_results'].append(results[0]['warning'])
                                    #print("box_data['test_results']: ", box_data['test_results'])
                                    percentage = (sum(box_data['test_results']) / len(box_data['test_results'])) * 100
                                    if percentage > 0:  # If there's any warning
                                        print(f"Wrong placement percentage: {percentage:.1f}% in {'Right' if box_idx == 0 else 'Left'} box")
                                        print(f"Similarity scores - Right: {results[0]['right_score']:.3f}, Left: {results[0]['left_score']:.3f}")
                                    
                                    test_duration_check = len(box_data['test_results']) >= int(comparer.TEST_DURATION / comparer.test_interval)
                                    #print(f"Test duration check: {test_duration_check}")
                                    #print(f"Current test results length: {len(box_data['test_results'])}")
                                    #print(f"Required length: {int(comparer.TEST_DURATION / comparer.test_interval)}")

                                    # Second condition check
                                    left_score_check = results[0]['left_score'] > comparer.warning_threshold
                                    right_score_check = results[0]['right_score'] > comparer.warning_threshold
                                    #print(f"Left score check: {left_score_check} ({results[box_idx]['left_score']} > {comparer.warning_threshold})")
                                    #print(f"Right score check: {right_score_check} ({results[box_idx]['right_score']} > {comparer.warning_threshold})")
                                    if len(box_data['test_results']) >= int(comparer.TEST_DURATION / comparer.test_interval) and (results[0]['left_score'] > comparer.warning_threshold or results[0]['right_score'] > comparer.warning_threshold):
                                        # Calculate if majority of tests showed wrong placement
                                        #print("CONDITIONS MET: Checking for wrong placement...")
                                        print("sum(box_data['test_results']): ", sum(box_data['test_results']))
                                        print("len(box_data['test_results']): ", len(box_data['test_results']))
                                        if (box_idx == 0 and results[0]['left_score'] > results[0]['right_score']) or \
                                            (box_idx == 1 and results[0]['right_score'] > results[0]['left_score']):
                                            print(f"WARNING: Wrong object placement detected in {'Right' if box_idx == 0 else 'Left'} box!")
                                            if box_idx == 0:
                                                comparer.right_box_color = 1 # red
                                            else:
                                                comparer.left_box_color = 1 # red
                                        if box_idx == 0:
                                            comparer.right_box_state = 1 # object processed & waiting for leaving
                                        else:
                                            comparer.left_box_state = 1 # object processed & waiting for leaving
                            else:
                                # Update previous bounding boxf
                                box_data['prev_bbox'] = (x1, y1, x2, y2)

                if (box_idx == 0 and comparer.right_box_state == 1 and comparer.is_right_box_empty) or (box_idx == 1 and comparer.left_box_state == 1 and comparer.is_left_box_empty):
                    if box_idx == 0:
                        comparer.right_box_state = 0
                        comparer.right_box_color = 0
                    else:
                        comparer.left_box_state = 0
                        comparer.left_box_color = 0
                        
                    # Reset tracking for this box
                    comparer.objects_in_boxes[box_idx] = {'object': None, 'start_time': 0, 'test_results': [], 'prev_bbox': None, 'bbox_history': deque(maxlen=comparer.BBOX_HISTORY_SIZE)}

            # Draw detection
            cv2.rectangle(comparer.frame_display, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

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

    comparer.cap.release()
    cv2.destroyAllWindows()
