import cv2
import numpy as np
from dilation import process_images_and_create_yolo_labels

def mse(imageA, imageB):
    """
    Calculate Mean Squared Error between two images
    Returns: MSE score (lower is better) and similarity percentage
    """
    # Ensure images are the same size
    if imageA.shape != imageB.shape:
        raise ValueError("Images must have the same dimensions")
    
    # Calculate MSE
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    # Convert MSE to similarity percentage (inverse relationship)
    # Using exponential scaling to get better distribution of values
    similarity = 100 * np.exp(-err / 1000)  # Adjust the scaling factor (1000) as needed
    
    return err, similarity

process_images_and_create_yolo_labels(["images/right_factory_0.png", "images/right_factory_1.png", "images/left_factory_0.png"], "yolo_labels")

# Load contour images (explicitly in grayscale)
img1 = cv2.imread("./images/right_factory_0.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("./images/right_factory_1.png", cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread("./images/left_factory_0.png", cv2.IMREAD_GRAYSCALE)

# Check if images were loaded successfully
if img1 is None or img2 is None or img3 is None:
    print("Error: Could not load one or more images")
    exit(1)


# Calculate MSE and similarity scores
mse_1_2, similarity_1_2_mse = mse(img1, img2)
mse_1_3, similarity_1_3_mse = mse(img1, img3)

print("\nImage Comparison Results:")
print("-" * 50)
print(f"Comparison between test image and base image:")
print(f"MSE Score: {mse_1_2:.2f} (lower is better)")
print(f"Similarity: {similarity_1_2_mse:.2f}%")
print("\nComparison between test image and different image:")
print(f"MSE Score: {mse_1_3:.2f} (lower is better)")
print(f"Similarity: {similarity_1_3_mse:.2f}%")