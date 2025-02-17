import cv2
import numpy as np

class BoundingBoxCamera:
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        self.boxes = [
            [(50, 200), (210, 360)],  # Right box
            [(350, 200), (510, 360)]   # Left box
        ]

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
        angles = [0, 90, 180, 270]
        
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
                
                # Load base images
                right_base = cv2.imread("right-base-image.png")
                left_base = cv2.imread("left-base-image.png")
                
                if right_base is not None and left_base is not None:
                    # Calculate similarities using template matching
                    right_score = self.get_best_template_match(current_crop, right_base)
                    left_score = self.get_best_template_match(current_crop, left_base)
                    
                    print(f"\nBox {i+1} Comparison Results:")
                    print("-" * 30)
                    print(f"Right base similarity: {right_score:.4f} (higher is better)")
                    print(f"Left base similarity: {left_score:.4f} (higher is better)")
                    
                    # Check if box is matching with wrong base image
                    if i == 0 and left_score > right_score and left_score > 0.8:
                        print("WARNING: Right box matches better with left base image!")
                    elif i == 1 and right_score > left_score and right_score > 0.8:
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
                    box_name = 'Right' if i == 0 else 'Left'
                    cv2.putText(frame_display, box_name, box[0], 
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
    cam = BoundingBoxCamera(camera_id=0)
    cam.run()