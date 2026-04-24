import cv2
import numpy as np
import tensorflow as tf
import json
import os

# ================= CONFIGURATION =================
MODEL_PATH = 'halmidi_model.h5'
LABELS_PATH = 'model_labels.json'

# 🔴 CHANGE THIS PATH BELOW TO TEST DIFFERENT IMAGES 🔴
# You can point this to any image file in your computer
TEST_IMAGE_PATH = 'test_data/SHH_TA/SHH_TA01.jpg'
# =================================================

# ================= KANNADA MAPPING =================
# This dictionary translates the "Folder Name" to "Modern Kannada"
# It matches exactly the 44 folders you created.
# ===================================================
# FINAL CORRECTED MAPPING (With Your Specific Edits)
# ===================================================
KANNADA_MAPPING = {
    # --- STANDARD ALPHABET (01-44) ---
    "01_a": "ಅ", "02_aa": "ಆ", "03_i": "ಇ", "04_ii": "ಈ", "05_u": "ಉ",
    "06_ru": "ಋ", "07_e": "ಎ", "08_oo": "ಓ", "09_am": "ಅಂ", "10_aha": "ಅಃ",

    "11_ka": "ಕ", "12_kha": "ಖ", "13_ga": "ಗ", "14_gha": "ಘ", "15_nga": "ಙ",
    "16_cha": "ಚ", "17_chha": "ಛ", "18_ja": "ಜ", "19_jha": "ಝ", "20_nya": "ಞ",
    "21_tta": "ಟ", "22_ttha": "ಠ", "23_dda": "ಡ", "24_ddha": "ಢ", "25_nna": "ಣ",
    "26_ta": "ತ", "27_tha": "ಥ", "28_da": "ದ", "29_dha": "ಧ", "30_na": "ನ",
    "31_pa": "ಪ", "32_pha": "ಫ", "33_ba": "ಬ", "34_bha": "ಭ", "35_ma": "ಮ",

    "36_ya": "ಯ", "37_ra": "ರ", "38_la": "ಲ", "39_va": "ವ", "40_sha": "ಶ",
    "41_ssha": "ಷ", "42_sa": "ಸ", "43_ha": "ಹ", "44_lla": "ಳ",

    # --- SPECIAL HALMIDI COMPOUNDS (45-67) ---
    "45_thi": "ತಿ (thi)",
    "46_shree": "ಶ್ರೀ (Shri)",
    "47_re": "ರಿ (ri)",
    "48_shh_va": "ಷ್ವ (shva)",
    
    # YOUR CORRECTIONS APPLIED BELOW:
    "49_gna_ga": "ಙ್ಗ (nga)",
    "50_sh_yaa": "ಶ್ಯಾ (shyaa)",
    "51_m_yaa": "ಮ್ಯಾ (myaa)",
    
    "52_ch_yu": "ಚ್ಯು (chyu)",
    "53_ta_aha": "ತಃ (tah)",
    "54_daa": "ದಾ (daa)",
    "55_vaa": "ವಾ (vaa)",
    
    "56_ksh_nnoo": "ಕ್ಷ್ಣೋ (kshno)",
    "57_ry_u": "ರ್ಯು (ryu)",  # Kept key as 'ry_uu' to match your folder name
    
    "58_gaa": "ಗಾ (gaa)",
    "59_n_taa": "ನ್ತಾ (nta)",
    "60_g_nihi": "ಗ್ನಿಃ (gnihi)",
    "61_shi": "ಶಿ (shi)",
    
    "62_shh_ttaa": "ಷ್ಟಾ (shta)",
    "63_naa": "ನಾ (naa)",
    "64_n_tu": "ನ್ತು (ntu)",
    
    "65_su": "ಸು (su)",
    "66_rsha": "ರ್ಶ (rsha)",
    "67_na_aha": "ನಃ (nah)"
}
# ===================================================

def predict_letter():
    print("\n🚀 Starting Prediction System...")

    # 1. Check if Model exists
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        print(f"❌ Critical Error: Missing '{MODEL_PATH}' or '{LABELS_PATH}'")
        print("   Did you run 'train_model.py' successfully?")
        return

    # 2. Load the AI Brain (Model)
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("🧠 Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 3. Load the Class Labels (The Dictionary)
    with open(LABELS_PATH, 'r') as f:
        class_indices = json.load(f)

    # 4. Load and Preprocess the Image
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"❌ Error: The image file '{TEST_IMAGE_PATH}' was not found.")
        return

    # Read Image as Grayscale
    img = cv2.imread(TEST_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("❌ Error: Could not open image. Is it a valid PNG/JPG?")
        return

    # Resize to 64x64 (Must match training size)
    img_resized = cv2.resize(img, (64, 64))
    
    # Normalize pixel values (0 to 1)
    img_final = img_resized / 255.0
    
    # Reshape: (1 image, 64 height, 64 width, 1 channel)
    img_final = np.reshape(img_final, (1, 64, 64, 1))

    # 5. Ask the AI to Predict
    prediction = model.predict(img_final)
    
    # Find the highest probability
    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # 6. Decode the Result
    # Get the folder name (e.g., "11_ka")
    folder_name = class_indices[str(predicted_index)]
    
    # Get the Modern Kannada Letter (e.g., "ಕ")
    kannada_char = KANNADA_MAPPING.get(folder_name, "Unknown Label")

    # 7. Print the Final Output
    print("\n" + "="*40)
    print(f"🖼️  INPUT IMAGE:    {TEST_IMAGE_PATH}")
    print("-" * 40)
    print(f"📂 DETECTED CLASS: {folder_name}")
    print(f"✅ KANNADA OUTPUT: {kannada_char}")
    print(f"📊 CONFIDENCE:     {confidence:.2f}%")
    print("="*40 + "\n")

if __name__ == '__main__':
    predict_letter()