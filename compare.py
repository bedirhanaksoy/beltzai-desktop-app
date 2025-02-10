import os
from pathlib import Path
import cv2
import numpy as np

# Define paths
base_image_path = "./images/right_0.jpeg"
test_image_path = "./images/right_1.jpeg"
output_dir = Path("./images/output")
output_dir.mkdir(parents=True, exist_ok=True)

def extract_and_save_contour(image_path, output_dir):
    """Extracts the largest contour after preprocessing and saves the dilated edges."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found or cannot be loaded: {image_path}")
        return None, None

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.medianBlur(gray_image, 5)

    # Apply thresholding
    _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Edge detection and dilation
    edges = cv2.Canny(thresh, 100, 200)
    edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # Save the dilated image
    output_path = os.path.join(output_dir, os.path.basename(image_path).replace(".png", "_dilated.png"))
    cv2.imwrite(output_path, edges_dilated)

    # Find contours
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"No object found in the image: {image_path}")
        return None, output_path

    return max(contours, key=cv2.contourArea), output_path  # Return the largest contour and saved image path

def compute_hu_moments_similarity(contour1, contour2):
    """Computes similarity using Hu Moments."""
    moments1 = cv2.HuMoments(cv2.moments(contour1)).flatten()
    moments2 = cv2.HuMoments(cv2.moments(contour2)).flatten()

    # Convert to log scale for better comparison
    log_moments1 = -np.sign(moments1) * np.log10(np.abs(moments1) + 1e-10)
    log_moments2 = -np.sign(moments2) * np.log10(np.abs(moments2) + 1e-10)

    # Compute Euclidean distance
    return np.linalg.norm(log_moments1 - log_moments2)

# Extract contours and save dilated images
contour_base, base_dilated_path = extract_and_save_contour(base_image_path, output_dir)
contour_test, test_dilated_path = extract_and_save_contour(test_image_path, output_dir)

# Flip the test image and extract contour from it
test_flipped_path = os.path.join(output_dir, "test_flipped.png")
test_image = cv2.imread(test_image_path)
test_flipped = cv2.flip(test_image, 1)  # Flip horizontally
cv2.imwrite(test_flipped_path, test_flipped)

# Extract contour from flipped test image
contour_test_flipped, test_flipped_dilated_path = extract_and_save_contour(test_flipped_path, output_dir)

# Compare contours
if contour_base is not None and contour_test is not None:
    match_score = compute_hu_moments_similarity(contour_base, contour_test)
    print(f"Hu Moments Match Score: {match_score}")

    if match_score < 0.1:
        result = "Correct object detected!"
    else:
        result = "Incorrect object or symmetric part detected!"

    # Compare with flipped version
    if contour_test_flipped is not None:
        match_score_flipped = compute_hu_moments_similarity(contour_base, contour_test_flipped)
        print(f"Flipped Hu Moments Match Score: {match_score_flipped}")

        if match_score_flipped < 0.1:
            result += " (Warning: Symmetric version matches, incorrect placement!)"
else:
    result = "Failed to extract contours from one or both images."

print(result)
