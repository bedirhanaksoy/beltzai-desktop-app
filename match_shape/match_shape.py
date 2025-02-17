import cv2
import numpy as np
from dilation import process_images_and_create_yolo_labels

def compare_shapes(img1, img2):
    """
    Compare shapes using contour matching
    Returns: similarity score (lower is better, 0 means perfect match)
    """
    # Find contours in both images
    contours1, _ = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Ensure contours were found
    if not contours1 or not contours2:
        raise ValueError("No contours found in one or both images")
    
    # Use the largest contour in each image
    cnt1 = max(contours1, key=cv2.contourArea)
    cnt2 = max(contours2, key=cv2.contourArea)
    
    # Compare shapes using cv2.CONTOURS_MATCH_I1 method
    similarity = cv2.matchShapes(cnt1, cnt2, cv2.CONTOURS_MATCH_I1, 0.0)
    
    # Convert to percentage (0 is perfect match, larger numbers indicate less similarity)
    # Using exponential scaling to convert to a 0-100 range
    similarity_percent = 100 * np.exp(-similarity * 10)  # Adjust scaling factor as needed
    
    return similarity, similarity_percent

# Process images
process_images_and_create_yolo_labels([
    "images/right_factory_0.png",
    "images/right_factory_1.png",
    "images/left_factory_0.png"
], "yolo_labels")

# Load images in grayscale
img1 = cv2.imread("./images/right_factory_0.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("./images/right_factory_1.png", cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread("./images/left_factory_0.png", cv2.IMREAD_GRAYSCALE)

# Check if images were loaded successfully
if img1 is None or img2 is None or img3 is None:
    print("Error: Could not load one or more images")
    exit(1)

# Compare shapes
score_1_2, similarity_1_2 = compare_shapes(img1, img2)
score_1_3, similarity_1_3 = compare_shapes(img1, img3)

# Print results
print("\nShape Comparison Results:")
print("-" * 50)
print(f"Comparison between test image and base image:")
print(f"Match Score: {score_1_2:.6f} (lower is better, 0 is perfect match)")
print(f"Similarity: {similarity_1_2:.2f}%")
print("\nComparison between test image and different image:")
print(f"Match Score: {score_1_3:.6f} (lower is better, 0 is perfect match)")
print(f"Similarity: {similarity_1_3:.2f}%")