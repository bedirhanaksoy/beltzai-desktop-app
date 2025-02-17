from PIL import Image
import imagehash

def get_best_similarity(image_path, compare_hashes):
    image = Image.open(image_path)
    best_phash_score = float('inf')
    best_dhash_score = float('inf')

    for angle in [0, 90, 180, 270]:
        rotated_image = image.rotate(angle)
        phash = imagehash.phash(rotated_image)
        dhash = imagehash.dhash(rotated_image)

        for compare_hash in compare_hashes:
            phash_score = phash - compare_hash['phash']
            dhash_score = dhash - compare_hash['dhash']

            if phash_score < best_phash_score:
                best_phash_score = phash_score
            if dhash_score < best_dhash_score:
                best_dhash_score = dhash_score

    return best_phash_score, best_dhash_score

# Load comparison images and calculate their hashes
hash2 = {
    'phash': imagehash.phash(Image.open("./images/right_factory_1.png")),
    'dhash': imagehash.dhash(Image.open("./images/right_factory_1.png"))
}
hash3 = {
    'phash': imagehash.phash(Image.open("./images/left_factory_0.png")),
    'dhash': imagehash.dhash(Image.open("./images/left_factory_0.png"))
}

# Get the best similarity scores for all rotations of right_factory_0.png
best_phash_score_2, best_dhash_score_2 = get_best_similarity("./images/right_factory_0.png", [hash2])
best_phash_score_3, best_dhash_score_3 = get_best_similarity("./images/right_factory_0.png", [hash3])

print("Best PHASH Hamming Distance for right_factory_1.png:", best_phash_score_2)  # Lower is better (0 = identical)
print("Best DHASH Hamming Distance for right_factory_1.png:", best_dhash_score_2)  # Lower is better (0 = identical)
print("Best PHASH Hamming Distance for left_factory_0.png:", best_phash_score_3)  # Lower is better (0 = identical)
print("Best DHASH Hamming Distance for left_factory_0.png:", best_dhash_score_3)  # Lower is better (0 = identical)

# Determine which image is more similar
if best_phash_score_2 < best_phash_score_3:
    print("right_factory_0.png is more similar to right_factory_1.png based on PHASH.")
else:
    print("right_factory_0.png is more similar to left_factory_0.png based on PHASH.")

if best_dhash_score_2 < best_dhash_score_3:
    print("right_factory_0.png is more similar to right_factory_1.png based on DHASH.")
else:
    print("right_factory_0.png is more similar to left_factory_0.png based on DHASH.")