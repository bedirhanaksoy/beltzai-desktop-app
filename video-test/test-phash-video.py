import cv2
import numpy as np
from PIL import Image
import imagehash
import io
from collections import deque
import time

class BoundingBoxCamera:
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        self.boxes = [
            [(50, 200), (250, 400)],  # Right box
            [(350, 200), (550, 400)]   # Left box
        ]
        self.frame_buffer = deque(maxlen=5)  # Store last 5 frames
        self.frame_display = None  # Initialize frame_display
        self.last_test_time = time.time()
        self.test_interval = 1.0  # Test every 1 second
        self.base_images_loaded = False
        self.right_hash = None
        self.left_hash = None

    def load_base_images(self):
        """Load base images and calculate their hashes"""
        try:
            self.right_hash = imagehash.phash(Image.open("right-base-image.png"))
            self.left_hash = imagehash.phash(Image.open("left-base-image.png"))
            self.base_images_loaded = True
            print("Base images loaded successfully")
        except Exception as e:
            print(f"Error loading base images: {e}")
            self.base_images_loaded = False

    def get_best_hash_similarity(self, image, base_image_hash):
        """Calculate best hash similarity across all rotations"""
        best_score = float('inf')
        success, buffer = cv2.imencode('.png', image)
        if not success:
            return float('inf')
        
        pil_image = Image.open(io.BytesIO(buffer))
        
        for angle in [0, 90, 180, 270]:
            rotated_image = pil_image.rotate(angle)
            current_hash = imagehash.phash(rotated_image)
            score = current_hash - base_image_hash
            best_score = min(best_score, score)
        
        return best_score

    def test_frame(self, frame):
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

                right_score = self.get_best_hash_similarity(current_crop, self.right_hash)
                left_score = self.get_best_hash_similarity(current_crop, self.left_hash)

                warning = (i == 0 and left_score < right_score) or (i == 1 and right_score < left_score)
                results.append({
                    'box': i+1,
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
            
            # Update box color in display
            if len(self.boxes) >= box_num:
                cv2.rectangle(self.frame_display, 
                            self.boxes[box_num-1][0], 
                            self.boxes[box_num-1][1], 
                            color, 2)
                
                # Print results to terminal
                print(f"\nBox {box_num} ({['Right', 'Left'][box_num-1]}):")
                print(f"Right base similarity: {result['right_score']:.1f}")
                print(f"Left base similarity: {result['left_score']:.1f}")
                if warning:
                    print("\033[91mWARNING: Possible mismatch detected!\033[0m")  # Red text
                else:
                    print("\033[92mMatch OK\033[0m")  # Green text

    def crop_and_save(self, box, filename):
        """Save cropped region as base image"""
        if box and len(box) == 2:
            x1, y1 = box[0]
            x2, y2 = box[1]
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            cropped = self.frame[y1:y2, x1:x2]
            cv2.imwrite(filename, cropped)
            print(f"Saved {filename}")
            self.load_base_images()  # Reload base images after saving

    def run(self):
        """Main loop for camera feed and interaction"""
        cv2.namedWindow('Camera Feed')

        while True:
            ret, self.frame = self.cap.read()
            if not ret:
                break

            self.frame_display = self.frame.copy()
            
            # Real-time testing
            current_time = time.time()
            if current_time - self.last_test_time >= self.test_interval:
                results = self.test_frame(self.frame)
                if results:
                    print("processing..")
                    self.process_and_display_results(results)
                self.last_test_time = current_time

            cv2.imshow('Camera Feed', self.frame_display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1') and len(self.boxes) >= 1:
                self.crop_and_save(self.boxes[0], "right-base-image.png")
            elif key == ord('2') and len(self.boxes) >= 2:
                self.crop_and_save(self.boxes[1], "left-base-image.png")
            elif key == ord('c'):
                self.boxes = []

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    cam = BoundingBoxCamera(camera_id=2)
    cam.load_base_images()
    cam.run()