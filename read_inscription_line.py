import cv2
import numpy as np
import tensorflow as tf
import json
import os

# ================= CONFIGURATION =================
MODEL_PATH = 'halmidi_model.h5'
LABELS_PATH = 'model_labels.json'
INPUT_LINE_IMAGE = 'test_lines/jayathi.jpeg' 
OUTPUT_DIR = 'segmented_output'

KANNADA_MAPPING = {
    "01_a": "ಅ", "02_aa": "ಆ", "03_i": "ಇ", "04_ii": "ಈ", "05_u": "ಉ", "06_ru": "ಋ", "07_e": "ಎ", "08_oo": "ಓ", "09_am": "ಅಂ", "10_aha": "ಅಃ", "11_ka": "ಕ", "12_kha": "ಖ", "13_ga": "ಗ", "14_gha": "ಘ", "15_nga": "ಙ", "16_cha": "ಚ", "17_chha": "ಛ", "18_ja": "ಜ", "19_jha": "ಝ", "20_nya": "ಞ", "21_tta": "ಟ", "22_ttha": "ಠ", "23_dda": "ಡ", "24_ddha": "ಢ", "25_nna": "ಣ", "26_ta": "ತ", "27_tha": "ಥ", "28_da": "ದ", "29_dha": "ಧ", "30_na": "ನ", "31_pa": "ಪ", "32_pha": "ಫ", "33_ba": "ಬ", "34_bha": "ಭ", "35_ma": "ಮ", "36_ya": "ಯ", "37_ra": "ರ", "38_la": "ಲ", "39_va": "ವ", "40_sha": "ಶ", "41_ssha": "ಷ", "42_sa": "ಸ", "43_ha": "ಹ", "44_lla": "ಳ", "45_thi": "ತಿ", "46_shree": "ಶ್ರೀ", "47_re": "ರಿ", "48_shh_va": "ಷ್ವ", "49_gna_ga": "ಙ್ಗ", "50_sh_yaa": "ಶ್ಯಾ", "51_m_yaa": "ಮ್ಯಾ", "52_ch_yu": "ಚ್ಯು", "53_ta_aha": "ತಃ", "54_daa": "ದಾ", "55_vaa": "ವಾ", "56_ksh_nnoo": "ಕ್ಷ್ಣೋ", "57_ry_uu": "ರ್ಯು", "58_gaa": "ಗಾ", "59_n_taa": "ನ್ತಾ", "60_g_nihi": "ಗ್ನಿಃ", "61_shi": "ಶಿ", "62_shh_ttaa": "ಷ್ಟಾ", "63_naa": "ನಾ", "64_n_tu": "ನ್ತು", "65_su": "ಸು", "66_rsha": "ರ್ಶ", "67_na_aha": "ನಃ"
}

def merge_vertical_boxes(boxes):
    """
    STRICT MERGE: Only merge if boxes are stacked vertically.
    """
    if not boxes: return []
    boxes.sort(key=lambda b: b[0])
    
    merged = []
    skip_indices = set()

    for i in range(len(boxes)):
        if i in skip_indices: continue
        
        current_box = boxes[i]
        x1, y1, w1, h1 = current_box
        cx1 = x1 + w1/2

        merged_box = current_box
        
        for j in range(i + 1, len(boxes)):
            if j in skip_indices: continue
            
            x2, y2, w2, h2 = boxes[j]
            cx2 = x2 + w2/2
            
            # --- THE FIX IS HERE ---
            # Reduced threshold from 20 to 8 pixels.
            # This ensures we don't merge side-by-side letters like Pa and Ri
            if abs(cx1 - cx2) < 8: 
                nx = min(x1, x2)
                ny = min(y1, y2)
                nw = max(x1+w1, x2+w2) - nx
                nh = max(y1+h1, y2+h2) - ny
                merged_box = (nx, ny, nw, nh)
                skip_indices.add(j)
                x1, y1, w1, h1 = merged_box 
                cx1 = x1 + w1/2

        merged.append(merged_box)
    return merged

def main():
    print("🧠 Loading AI Model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, 'r') as f:
        class_indices = json.load(f)

    print(f"📖 Reading: {INPUT_LINE_IMAGE}")
    line_image = cv2.imread(INPUT_LINE_IMAGE)
    
    # Visualization
    SCALE = 3
    h, w = line_image.shape[:2]
    display_image = cv2.resize(line_image, (w * SCALE, h * SCALE), interpolation=cv2.INTER_NEAREST)

    # Preprocessing
    gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Erosion to separate touching letters
    kernel = np.ones((2,2), np.uint8)
    eroded_thresh = cv2.erode(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(eroded_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    raw_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 5 and h > 10:
            raw_boxes.append((x, y, w, h))

    # Apply Strict Vertical Merge
    final_boxes = merge_vertical_boxes(raw_boxes)
    final_boxes.sort(key=lambda b: b[0])

    print(f"\n✅ Found {len(raw_boxes)} parts. Merged into {len(final_boxes)} characters.")

    final_text_str = ""
    for i, box in enumerate(final_boxes):
        x, y, w, h = box
        
        # Crop & Predict
        char_crop = thresh[y:y+h, x:x+w]
        char_crop = cv2.bitwise_not(char_crop)
        padded_char = cv2.copyMakeBorder(char_crop, 10,10,10,10, cv2.BORDER_CONSTANT, value=[255,255,255])
        
        img_resized = cv2.resize(padded_char, (64, 64)) / 255.0
        img_final = np.reshape(img_resized, (1, 64, 64, 1))
        
        prediction = model.predict(img_final, verbose=0)
        idx = np.argmax(prediction)
        
        folder = class_indices[str(idx)]
        label_text = folder.split('_')[-1]
        k_char = KANNADA_MAPPING.get(folder, "?").split(" ")[0]
        final_text_str += k_char + " "

        # Draw
        sx, sy, sw, sh = x*SCALE, y*SCALE, w*SCALE, h*SCALE
        cv2.rectangle(display_image, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
        cv2.putText(display_image, label_text, (sx, sy + sh + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        print(f"[{i}] {folder} -> {k_char}")

    print("="*50)
    print(f"▶️  TEXT: {final_text_str}")
    print("="*50)

    cv2.imshow('Analysis', display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()