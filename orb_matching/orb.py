import cv2
import numpy as np
from dilation import process_images_and_create_yolo_labels

def match_images(img1_path, img2_path):
    # Load images
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        raise FileNotFoundError("Could not load one or both images")

    # Create ORB detector
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Calculate metrics
    similarity_score = len(matches) / max(len(kp1), len(kp2))
    average_distance = np.mean([match.distance for match in matches])
    
    return similarity_score, average_distance

# Process images
process_images_and_create_yolo_labels(["images/right_factory_0.png", "images/right_factory_1.png", "images/left_factory_0.png"], "yolo_labels")

# Match images
score_1_2, distance_1_2 = match_images("./bounding_box_images/right_factory_0.png", "./bounding_box_images/right_factory_1.png")
score_1_3, distance_1_3 = match_images("./bounding_box_images/right_factory_0.png", "./bounding_box_images/left_factory_0.png")

# Normalize and print results
print(f"ORB Similarity Score between img1 and img2: {score_1_2 * 100:.2f}%")
print(f"Average Distance between img1 and img2: {distance_1_2:.2f}")

print(f"ORB Similarity Score between img1 and img3: {score_1_3 * 100:.2f}%")
print(f"Average Distance between img1 and img3: {distance_1_3:.2f}")