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

# For testing purposes
test_video_path = str(resources_path / "test_video/test_video.webm")

class CustomPartTracker:
    def __init__(self, camera_id=0, model_path=models_path):
        
        self.cap = cv2.VideoCapture(test_video_path)

        self.boxes = [
            [(50, 200), (230, 380)],  # Right box
            [(350, 200), (530, 380)]   # Left box
        ]
        # Initialize YOLO model
        self.model = YOLO(model_path)
        
        # Add these parameters
        self.BBOX_HISTORY_SIZE = 5  # Number of previous bounding boxes to store
        self.MOVEMENT_THRESHOLD = 5  # Maximum allowed movement in pixels

        # Track objects in boxes
        self.objects_in_boxes = {
            0: {'object': None, 'start_time': 0, 'test_results': [], 'prev_bbox': None, 'bbox_history': deque(maxlen=self.BBOX_HISTORY_SIZE)},
            1: {'object': None, 'start_time': 0, 'test_results': [], 'prev_bbox': None, 'bbox_history': deque(maxlen=self.BBOX_HISTORY_SIZE)}
        }
        
        self.left_box_state = 0 # state 0: waiting for an object, state 1: object processed & waiting for leaving
        self.right_box_state = 0 # state 0: waiting for an object, state 1: object processed & waiting for leaving

        self.right_box_color = 0 # 0: green, 1: red
        self.left_box_color = 0 # 0: green, 1: red

        self.is_left_box_empty = True
        self.is_right_box_empty = True

        self.STILL_THRESHOLD = 0.2  # Time threshold for considering object still (seconds)
        self.TEST_DURATION = 0.2    # Duration for running tests (seconds)
        self.warning_threshold = 0.8  # Threshold for warning if similarity score is below this value

        self.frame_buffer = deque(maxlen=5)
        self.frame_display = None
        self.last_test_time = time.time()
        self.test_interval = 1/20  # Adjust Testing frame
        self.base_images_loaded = False
        self.right_base = None
        self.left_base = None
        self.yolo_detections = None

    def load_base_images(self):
        """Load base images"""
        try:
            self.right_base = cv2.imread(right_base_image_path)
            self.left_base = cv2.imread(left_base_image_path)
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
        #print("box_idx: ", box_idx)
        if not self.base_images_loaded:
            return None
        results = []
        box = self.boxes[box_idx]
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
        warning = ((box_idx == 0 and left_score > right_score) and left_score > self.warning_threshold ) or \
                ((box_idx == 1 and right_score > left_score) and right_score > self.warning_threshold)
        results.append({
            'box': box_idx,
            'right_score': right_score,
            'left_score': left_score,
            'warning': warning
        })
        #print("results: ", results)
        return results

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

    def check_if_box_is_empty(self, detections):
        """
        Check if any detected object is inside the specified box
        
        Args:
            box (list): Box coordinates in format [(x1,y1), (x2,y2)]
            detections (tensor): YOLO detection results containing bounding boxes
            
        Returns:
            bool: True if any object is detected in the box, False otherwise
        """
        box_x1, box_y1 = self.boxes[0][0]
        box_x2, box_y2 = self.boxes[0][1]
        box_x3, box_y3 = self.boxes[1][0]
        box_x4, box_y4 = self.boxes[1][1]

        boxes_empty_status = [True, True]

        for det in detections:
            # Extract coordinates and confidence
            x1, y1, x2, y2, conf, cls = det
            
            # Skip low confidence detections
            if conf < 0.5:
                continue
                
            # Calculate object's center point
            obj_center_x = (x1 + x2) / 2
            obj_center_y = (y1 + y2) / 2
            
            # Check if object's center is inside the box
            if (box_x1 <= obj_center_x <= box_x2 and 
                box_y1 <= obj_center_y <= box_y2):
                boxes_empty_status[0] = False
            
            if (box_x3 <= obj_center_x <= box_x4 and 
                box_y3 <= obj_center_y <= box_y4):
                boxes_empty_status[1] = False
                
        return boxes_empty_status[0], boxes_empty_status[1]

    def check_if_object_stable(self, current_bbox, bbox_history):
        """
        Check if object has been stable across multiple frames
        
        Args:
            current_bbox: tuple (x1, y1, x2, y2) of current detection
            bbox_history: deque of previous bounding boxes
        
        Returns:
            bool: True if object is stable across frames
        """
        if len(bbox_history) < self.BBOX_HISTORY_SIZE:
            return False
            
        for prev_bbox in bbox_history:
            if not (abs(prev_bbox[0] - current_bbox[0]) < self.MOVEMENT_THRESHOLD and
                    abs(prev_bbox[1] - current_bbox[1]) < self.MOVEMENT_THRESHOLD and
                    abs(prev_bbox[2] - current_bbox[2]) < self.MOVEMENT_THRESHOLD and
                    abs(prev_bbox[3] - current_bbox[3]) < self.MOVEMENT_THRESHOLD):
                return False
        return True

if __name__ == "__main__":
    cam = CustomPartTracker(camera_id=0, model_path=models_path)
    cam.load_base_images()
    cam.run()