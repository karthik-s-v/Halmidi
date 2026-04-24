import cv2
import numpy as np

# Path to your stone image
IMAGE_PATH = 'test_data/01_ja.png' # Change this to your image name

# 1. Read the image
img = cv2.imread(IMAGE_PATH)

if img is None:
    print("Error: Image not found!")
else:
    # 2. Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Apply Thresholding (The Magic Step)
    # This turns dark pixels (shadows) to Black and light pixels (stone) to White
    # We use 'Binary Inverted' because usually text is darker than stone
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 4. Save and Show
    cv2.imwrite('processed_stone.png', binary)
    print("✅ Processed image saved as 'processed_stone.png'. Check it!")