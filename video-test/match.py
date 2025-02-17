import cv2
import numpy as np

def rotate_image(image, angle):
    h, w = image.shape
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated_image

# Load images in grayscale
base_image_0 = cv2.imread("./images/right_factory_1.png", cv2.IMREAD_GRAYSCALE)
base_image_1 = cv2.imread("./images/left_factory_0.png", cv2.IMREAD_GRAYSCALE)
test_image = cv2.imread("./images/factory_0.png", cv2.IMREAD_GRAYSCALE)

# Define angles to rotate
angles = [0, 90, 180, 270]

best_match = None
best_score = -1
best_base_image = None

for base_image, base_name in zip([base_image_0, base_image_1], ["Base 0", "Base 1"]):
    for angle in angles:
        rotated_base = rotate_image(base_image, angle)
        result = cv2.matchTemplate(test_image, rotated_base, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        if max_val > best_score:
            best_score = max_val
            best_match = rotated_base
            best_base_image = base_name

print(f"The best match is with {best_base_image} with a similarity score of {best_score:.4f}")

cv2.imshow("Best Match", best_match)
cv2.destroyAllWindows()
