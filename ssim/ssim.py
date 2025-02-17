import cv2
from skimage.metrics import structural_similarity as ssim

def compare_images(image1, image2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    score, _ = ssim(gray1, gray2, full=True)
    return score  # 1.0 means identical, lower means more difference

base_img1 = cv2.imread("./images/right_factory_1.png")
base_img2 = cv2.imread("./images/left_factory_0.png")
test_img3 = cv2.imread("./images/right_factory_0.png")

score1_3 = compare_images(base_img1, test_img3)
score2_3 = compare_images(base_img2, test_img3)

print("SSIM Score between img1 and img3:", score1_3)
print("SSIM Score between img2 and img3:", score2_3)

if score1_3 >= score2_3:
    print("img1 and img3 are the most similar.")
else:
    print("img2 and img3 are the most similar.")
