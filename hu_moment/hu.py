import cv2
import numpy as np
from dilation import process_images_and_create_yolo_labels

# Compute Hu Moments
def get_hu_moments(img):
    # Convert to grayscale if image is color
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(img)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments

#process_images_and_create_yolo_labels(["images/right_factory_0.png", "images/right_factory_1.png", "images/left_factory_0.png"], "yolo_labels")

# Load contour images (explicitly in grayscale)
img1 = cv2.imread("./images/right_factory_0.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("./images/right_factory_1.png", cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread("./images/left_factory_0.png", cv2.IMREAD_GRAYSCALE)

# Check if images were loaded successfully
if img1 is None or img2 is None or img3 is None:
    print("Error: Could not load one or more images")
    exit(1)

hu1 = get_hu_moments(img1)
hu2 = get_hu_moments(img2)
hu3 = get_hu_moments(img3)

# Compute similarity using Euclidean Distance
distance_1_2 = np.linalg.norm(hu1 - hu2)
distance_1_3 = np.linalg.norm(hu1 - hu3)

# Use exponential scaling for better handling of small distances
def calculate_similarity(distance, sensitivity=50.0):
    """
    Calculate similarity using exponential scaling
    sensitivity: controls how quickly similarity drops with distance (higher = more sensitive)
    """
    return np.exp(-sensitivity * distance) * 100

# Calculate similarities
similarity_1_2_percent = calculate_similarity(distance_1_2)
similarity_1_3_percent = calculate_similarity(distance_1_3)

# Print results
print(f"Hu Moments Distance between img1 and img2: {distance_1_2:.6f}")
print(f"Similarity between img1 and img2: {similarity_1_2_percent:.2f}%")

print(f"Hu Moments Distance between img1 and img3: {distance_1_3:.6f}")
print(f"Similarity between img1 and img3: {similarity_1_3_percent:.2f}%")