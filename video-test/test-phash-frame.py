import cv2
import numpy as np
from PIL import Image
import imagehash
import io

class BoundingBoxCamera:
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        self.boxes = [
            [(50, 200), (250, 400)],  # Right box
            [(350, 200), (550, 400)]   # Left box
        ]

    def get_best_hash_similarity(self, image, base_image_hash):
        """Calculate best hash similarity across all rotations"""
        best_score = float('inf')
        
        # Convert OpenCV image to PIL format
        success, buffer = cv2.imencode('.png', image)
        if not success:
            return float('inf')
        
        pil_image = Image.open(io.BytesIO(buffer))
        
        # Check similarity for all rotations
        for angle in [0, 90, 180, 270]:
            rotated_image = pil_image.rotate(angle)
            current_hash = imagehash.phash(rotated_image)
            score = current_hash - base_image_hash
            best_score = min(best_score, score)
            
        return best_score

    def test_boxes(self):
        """Test current bounding boxes against saved base images"""
        for i, box in enumerate(self.boxes):
            if len(box) == 2:
                # Crop current frame
                x1, y1 = box[0]
                x2, y2 = box[1]
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                current_crop = self.frame[y1:y2, x1:x2]
                
                # Load base images and calculate their hashes
                right_base = cv2.imread("right-base-image.png")
                left_base = cv2.imread("left-base-image.png")
                
                if right_base is not None and left_base is not None:
                    # Convert base images to PIL and calculate hashes
                    right_hash = imagehash.phash(Image.open("right-base-image.png"))
                    left_hash = imagehash.phash(Image.open("left-base-image.png"))
                    
                    # Calculate similarities
                    right_score = self.get_best_hash_similarity(current_crop, right_hash)
                    left_score = self.get_best_hash_similarity(current_crop, left_hash)
                    
                    print(f"\nBox {i+1} Comparison Results:")
                    print("-" * 30)
                    print(f"Right base similarity: {right_score} (lower is better)")
                    print(f"Left base similarity: {left_score} (lower is better)")
                    
                    # Check if box is matching with wrong base image
                    if i == 0 and left_score < right_score:
                        print("WARNING: Right box matches better with left base image!")
                    elif i == 1 and right_score < left_score:
                        print("WARNING: Left box matches better with right base image!")
                else:
                    print(f"Could not load base images for Box {i+1}")

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

    def run(self):
        """Main loop for camera feed and interaction"""
        cv2.namedWindow('Camera Feed')

        while True:
            ret, self.frame = self.cap.read()
            if not ret:
                break

            frame_display = self.frame.copy()
            for i, box in enumerate(self.boxes):
                if len(box) == 2:
                    cv2.rectangle(frame_display, box[0], box[1], (0, 255, 0), 2)
                    cv2.putText(frame_display, f'Box {i+1}', box[0], 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow('Camera Feed', frame_display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1') and len(self.boxes) >= 1:
                self.crop_and_save(self.boxes[0], "right-base-image.png")
            elif key == ord('2') and len(self.boxes) >= 2:
                self.crop_and_save(self.boxes[1], "left-base-image.png")
            elif key == ord('t'):
                self.test_boxes()
            elif key == ord('c'):
                self.boxes = []

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    cam = BoundingBoxCamera(camera_id=2)
    cam.run()
