import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# ================= CONFIGURATION =================
INPUT_DIR = 'Raw_Data'        # Where your 1 crop is
OUTPUT_DIR = 'Dataset_Final'  # Where the 100 images will go
IMAGES_PER_CLASS = 100        # How many images to generate per letter
IMG_SIZE = (64, 64)           # Standard size for the AI (Do not change)
# =================================================

# 1. Define the Augmentation Logic (The "Variations")
datagen = ImageDataGenerator(
    rotation_range=10,        # Rotate slightly (text isn't always straight)
    width_shift_range=0.1,    # Move left/right
    height_shift_range=0.1,   # Move up/down
    shear_range=0.1,          # Slant the text slightly
    zoom_range=0.1,           # Zoom in/out slightly
    fill_mode='nearest',      # Fill empty pixels after rotation
    brightness_range=[0.8, 1.2] # Make it slightly darker/lighter
)

# 2. Create Output Directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"🚀 Starting Data Augmentation from {INPUT_DIR}...")

# 3. Process Each Folder
classes = os.listdir(INPUT_DIR)

for class_name in classes:
    class_path = os.path.join(INPUT_DIR, class_name)
    
    # Skip if it's not a folder
    if not os.path.isdir(class_path):
        continue

    # Create destination folder
    save_path = os.path.join(OUTPUT_DIR, class_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # List all images in the raw folder
    images = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(images) == 0:
        print(f"⚠️ Warning: No images found in {class_name}")
        continue

    print(f"Processing Class: {class_name}...")

    # Load the single image (or few images)
    for img_name in images:
        img_path = os.path.join(class_path, img_name)
        
        # Load and Preprocess
        img = load_img(img_path, target_size=IMG_SIZE, color_mode='grayscale') 
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape) # Reshape for the generator

        # Generate variations
        i = 0
        for batch in datagen.flow(x, batch_size=1, 
                                  save_to_dir=save_path, 
                                  save_prefix='aug', 
                                  save_format='png'):
            i += 1
            if i >= IMAGES_PER_CLASS:
                break  # Stop after generating 100 images

print(f"\n✅ SUCCESS! Dataset generated in '{OUTPUT_DIR}' folder.")
print("You are now ready for Phase 2 (Training).")