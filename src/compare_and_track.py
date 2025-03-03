import cv2
import numpy as np
from collections import deque
import time
from ultralytics import YOLO
from pathlib import Path

MAIN_PATH = Path(__file__).resolve()
resources_path = MAIN_PATH.resolve().parent.parent / "resources"

models_path = resources_path / "models/right_part_medium.pt"
base_image_path = resources_path / "base_images"

# For testing purposes
test_video_path = resources_path / "test_video/test_video.webm" 

class CustomPartTracker:
    def __init__(self, camera_id=0, model_path=models_path):
        
        self.cap = cv2.VideoCapture(str(test_video_path))
        # self.boxes = [
        #    [(50, 200), (200, 350)],  # Right box
        #    [(350, 200), (500, 350)]   # Left box
        #] 
        self.boxes = [
            [(50, 200), (230, 380)],  # Right box
            [(350, 200), (530, 380)]   # Left box
        ]
        # Initialize YOLO model
        self.model = YOLO(model_path)
        
        # Track objects in boxes
        self.objects_in_boxes = {
            0: {'object': None, 'start_time': 0, 'test_results': [], 'prev_bbox': None},  # Right box
            1: {'object': None, 'start_time': 0, 'test_results': [], 'prev_bbox': None}   # Left box
        }
        
        self.left_box_state = 0 # state 0: waiting for an object, state 1: object processed & waiting for leaving
        self.right_box_state = 0 # state 0: waiting for an object, state 1: object processed & waiting for leaving

        self.STILL_THRESHOLD = 0.0  # Time threshold for considering object still (seconds)
        self.TEST_DURATION = 0.2    # Duration for running tests (seconds)
        self.warning_threshold = 0.8  # Threshold for warning if similarity score is below this value

        self.frame_buffer = deque(maxlen=5)
        self.frame_display = None
        self.last_test_time = time.time()
        self.test_interval = 1/10  # Adjust Testing frame
        self.base_images_loaded = False
        self.right_base = None
        self.left_base = None

    def load_base_images(self):
        """Load base images"""
        try:
            self.right_base = cv2.imread(str(base_image_path / "right-base-image.png"))
            self.left_base = cv2.imread(str(base_image_path / "left-base-image.png"))
            self.base_images_loaded = True
            print("Base images loaded successfully")
        except Exception as e:
            print(f"Error loading base images: {e}")
            self.base_images_loaded = False

    def rotate_image(self, image, angle):
        """Rotate image by given angle"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
        return rotated_image

    def get_best_template_match(self, image, template):
        """Calculate best template matching score across all rotations"""
        best_score = -1
        angles = range(0, 360, 15)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(template.shape) == 3:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        for angle in angles:
            rotated_template = self.rotate_image(template, angle)
            result = cv2.matchTemplate(image, rotated_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            best_score = max(best_score, max_val)
        
        return best_score

    def test_frame(self, frame, box_idx):
        """Test current frame and return results"""
        if not self.base_images_loaded:
            return None
        results = []
        for i, box in enumerate(self.boxes):
            if len(box) == 2:
                x1, y1 = box[0]
                x2, y2 = box[1]
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                current_crop = frame[y1:y2, x1:x2]
                # Compute symmetric versions of base images
                right_base_symmetric = cv2.flip(self.right_base, 1)
                left_base_symmetric = cv2.flip(self.left_base, 1)
                if box_idx == 0:
                    right_score = self.get_best_template_match(current_crop, self.right_base)
                    left_score = self.get_best_template_match(current_crop, right_base_symmetric)
                else:
                    right_score = self.get_best_template_match(current_crop, left_base_symmetric)
                    left_score = self.get_best_template_match(current_crop, self.left_base)
                warning = ((i == 0 and left_score > right_score) and left_score > self.warning_threshold ) or \
                        ((i == 1 and right_score > left_score) and right_score > self.warning_threshold)
                results.append({
                    'box': i + 1,
                    'right_score': right_score,
                    'left_score': left_score,
                    'warning': warning
                })
        return results

    def process_and_display_results(self, results):
        """Process and display test results"""
        if not results:
            return

        print("\033[2J\033[H")  # Clear terminal
        print("Real-time Comparison Results:")
        print("-" * 50)
        
        for result in results:
            box_num = result['box']
            warning = result['warning']
            color = (0, 0, 255) if warning else (0, 255, 0)  # Red if warning, green if ok
            
            if len(self.boxes) >= box_num:
                # Draw box
                box = self.boxes[box_num-1]
                cv2.rectangle(self.frame_display, 
                            box[0], 
                            box[1], 
                            color, 2)
                
                # Add text label above box
                label = "Right Box" if box_num == 1 else "Left Box"
                text_x = box[0][0]
                text_y = box[0][1] - 10  # 10 pixels above the box
                cv2.putText(self.frame_display, label, 
                           (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, color, 2)
                
                # Calculate and display box area
                width = abs(box[1][0] - box[0][0])
                height = abs(box[1][1] - box[0][1])
                area = width * height
                area_text = f"Area: {area}px"
                cv2.putText(self.frame_display, area_text,
                           (text_x, text_y - 25),  # 25 pixels above the label
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, color, 2)
                
                print(f"\nBox {box_num} ({['Right', 'Left'][box_num-1]}):")
                print(f"Box Area: {area} pixels")
                print(f"Right base similarity: {result['right_score']:.4f}")
                print(f"Left base similarity: {result['left_score']:.4f}")
                if warning:
                    print("\033[91mWARNING: Possible mismatch detected!\033[0m")
                else:
                    print("\033[92mMatch OK\033[0m")

    def crop_and_save(self, box, filename):
        """Save cropped region as base image with an additional 30 pixels margin"""
        if box and len(box) == 2:
            x1, y1 = box[0]
            x2, y2 = box[1]
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # Add 80 pixels margin
            x1 = max(0, x1 - 65)
            y1 = max(0, y1 - 65)
            x2 = min(self.frame.shape[1], x2 + 65)
            y2 = min(self.frame.shape[0], y2 + 65)
            
            cropped = self.frame[y1:y2, x1:x2]
            cv2.imwrite(filename, cropped)
            print(f"Saved {filename}")
            self.load_base_images()  # Reload base images after saving

    def check_object_in_box(self, box_coords, object_bbox):
        """Check if detected object is inside a test box"""
        box_x1, box_y1 = box_coords[0]
        box_x2, box_y2 = box_coords[1]
        obj_x1, obj_y1, obj_x2, obj_y2 = object_bbox
        
        # Check if object's center point is inside the box
        obj_center_x = (obj_x1 + obj_x2) / 2
        obj_center_y = (obj_y1 + obj_y2) / 2
        
        return (box_x1 <= obj_center_x <= box_x2 and 
                box_y1 <= obj_center_y <= box_y2)

    def run(self):
        """Main loop for camera feed and interaction"""
        cv2.namedWindow('Camera Feed')

        while True:
            ret, self.frame = self.cap.read()
            if not ret:
                break

            self.frame_display = self.frame.copy()
            
            # Draw boxes and labels
            for i, box in enumerate(self.boxes):
                # Get box coordinates
                x1, y1 = box[0]
                x2, y2 = box[1]
                
                # Draw box
                cv2.rectangle(self.frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label
                label = "Right Box" if i == 0 else "Left Box"
                # Position the label above the box
                label_x = x1
                label_y = y1 - 10
                cv2.putText(self.frame_display, label, 
                           (label_x, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 255, 0), 2)
                
                # Calculate and display area
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                area = width * height
                area_text = f"Area: {area}px"
                cv2.putText(self.frame_display, area_text,
                           (label_x, label_y - 25),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, (0, 255, 0), 2)
            
            # Run YOLO detection
            results = self.model(self.frame, verbose=False)
            current_time = time.time()
            
            # Process each detection
            for det in results[0].boxes.data:
                x1, y1, x2, y2, conf, cls = det
                if conf < 0.5:  # Confidence threshold
                    continue
                    
                # Check for each test box
                for box_idx, box in enumerate(self.boxes):
                    if (box_idx == 0 and self.right_box_state == 0) or (box_idx == 1 and self.left_box_state == 0):
                        if self.check_object_in_box(box, (x1, y1, x2, y2)):
                            box_data = self.objects_in_boxes[box_idx]
                            
                            # New object detected in box
                            if box_data['object'] is None:
                                box_data['object'] = cls.item()
                                box_data['start_time'] = current_time
                                box_data['test_results'] = []
                                box_data['prev_bbox'] = (x1, y1, x2, y2)
                            
                            # Check if object has been still for threshold time
                            if (current_time - box_data['start_time'] >= self.STILL_THRESHOLD and 
                                len(box_data['test_results']) < int(self.TEST_DURATION / self.test_interval)):
                                
                                # Compare current bounding box with previous bounding box
                                prev_bbox = box_data['prev_bbox']
                                if prev_bbox and (abs(prev_bbox[0] - x1) < 5 and abs(prev_bbox[1] - y1) < 5 and
                                                abs(prev_bbox[2] - x2) < 5 and abs(prev_bbox[3] - y2) < 5):
                                    # Run template matching test
                                    results = self.test_frame(self.frame, box_idx=box_idx)
                                    if results:
                                        box_data['test_results'].append(results[box_idx]['warning'])
                                        percentage = (sum(box_data['test_results']) / len(box_data['test_results'])) * 100
                                        if percentage > 0:  # If there's any warning
                                            print(f"Wrong placement percentage: {percentage:.1f}% in {'Right' if box_idx == 0 else 'Left'} box")
                                            print(f"Similarity scores - Right: {results[box_idx]['right_score']:.3f}, Left: {results[box_idx]['left_score']:.3f}")
                                        
                                        # If we have enough test results
                                        if len(box_data['test_results']) >= int(self.TEST_DURATION / self.test_interval):
                                            # Calculate if majority of tests showed wrong placement
                                            wrong_placement = sum(box_data['test_results']) > len(box_data['test_results']) / 2
                                            if wrong_placement: # INITIALIZATION ALGORITHM STOP OR START LOGIC WILL BE IMPLEMENTED HERE
                                                print(f"WARNING: Wrong object placement detected in {'Right' if box_idx == 0 else 'Left'} box!")
                                                # Add the left/right attribute to the object class
                                                # Here you would update your YOLO model's class attributes
                                            if box_idx == 0:
                                                self.right_box_state = 1
                                            else:
                                                self.left_box_state = 1
                                            
                                            # Reset tracking for this box
                                            #box_data['object'] = None
                                            box_data['test_results'] = []
                                            box_data['prev_bbox'] = None
                                else:
                                    # Update previous bounding boxf
                                    box_data['prev_bbox'] = (x1, y1, x2, y2)

                    if (box_idx == 0 and self.right_box_state == 1) or (box_idx == 1 and self.left_box_state == 1):
                        if not self.check_object_in_box(box, (x1, y1, x2, y2)):
                            if box_idx == 0:
                                self.right_box_state = 0
                            else:
                                self.left_box_state = 0

                # Draw detection
                cv2.rectangle(self.frame_display, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Display frame
            cv2.imshow('Camera Feed', self.frame_display)

            # Handle key events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1') and len(self.boxes) >= 1:
                self.crop_and_save(self.boxes[0], str(base_image_path / "right-base-image.png"))
            elif key == ord('2') and len(self.boxes) >= 2:
                self.crop_and_save(self.boxes[1], str(base_image_path / "left-base-image.png"))
            elif key == ord('c'):
                self.boxes = []

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    cam = CustomPartTracker(camera_id=2, model_path=str(models_path))
    cam.load_base_images()
    cam.run()