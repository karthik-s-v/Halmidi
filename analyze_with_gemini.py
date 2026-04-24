import cv2
import numpy as np
import tensorflow as tf
import json
import google.genai as genai

# ================= CONFIG =================
GOOG_API_KEY = "AIzaSyDk1jFQjeo-56JrxWjYaRwhGSL6um3yOrg"

MODEL_PATH = "halmidi_model.h5"
LABELS_PATH = "model_labels.json"

# ================= KANNADA MAP =================
KANNADA_MAPPING = {
    "01_a": "ಅ",
    "02_aa": "ಆ",
    "03_i": "ಇ",
    "04_ii": "ಈ",
    "05_u": "ಉ",
    "06_ru": "ಋ",
    "07_e": "ಎ",
    "08_oo": "ಓ",
    "09_am": "ಅಂ",
    "10_aha": "ಅಃ",

    "11_ka": "ಕ",
    "12_kha": "ಖ",
    "13_ga": "ಗ",
    "14_gha": "ಘ",
    "15_nga": "ಙ",

    "16_cha": "ಚ",
    "17_chha": "ಛ",
    "18_ja": "ಜ",
    "19_jha": "ಝ",
    "20_nya": "ಞ",

    "21_tta": "ಟ",
    "22_ttha": "ಠ",
    "23_dda": "ಡ",
    "24_ddha": "ಢ",
    "25_nna": "ಣ",

    "26_ta": "ತ",
    "27_tha": "ಥ",
    "28_da": "ದ",
    "29_dha": "ಧ",
    "30_na": "ನ",

    "31_pa": "ಪ",
    "32_pha": "ಫ",
    "33_ba": "ಬ",
    "34_bha": "ಭ",
    "35_ma": "ಮ",

    "36_ya": "ಯ",
    "37_ra": "ರ",
    "38_la": "ಲ",
    "39_va": "ವ",

    "40_sha": "ಶ",
    "41_ssha": "ಷ",
    "42_sa": "ಸ",
    "43_ha": "ಹ",
    "44_lla": "ಳ",

    "45_thi": "ತಿ",
    "46_shree": "ಶ್ರೀ",
    "47_re": "ರಿ",

    "48_shh_va": "ಷ್ವ",
    "49_gna_ga": "ಙ್ಗ",
    "50_sh_yaa": "ಶ್ಯಾ",
    "51_m_yaa": "ಮ್ಯಾ",
    "52_ch_yu": "ಚ್ಯು",

    "53_ta_aha": "ತಃ",
    "54_daa": "ದಾ",
    "55_vaa": "ವಾ",

    "56_ksh_nnoo": "ಕ್ಷ್ಣೋ",
    "57_ry_u": "ರ್ಯು",
    "58_gaa": "ಗಾ",

    "59_n_taa": "ನ್ತಾ",
    "60_g_nihi": "ಗ್ನಿಃ",
    "61_shi": "ಶಿ",
    "62_shh_ttaa": "ಷ್ಟಾ",
    "63_naa": "ನಾ",

    "64_n_tu": "ನ್ತು",
    "65_su": "ಸು",
    "66_rsha": "ರ್ಶ",
    "67_na_aha": "ನಃ"
}


# ================= LOAD MODEL ONCE =================
cnn_model = tf.keras.models.load_model(MODEL_PATH)
cnn_labels = json.load(open(LABELS_PATH, "r", encoding="utf-8"))

# ================= OCR =================
def perform_ocr(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return ""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    text = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 10 or h < 10:
            continue

        char = thresh[y:y+h, x:x+w]
        char = cv2.bitwise_not(char)
        char = cv2.copyMakeBorder(
            char, 10, 10, 10, 10,
            cv2.BORDER_CONSTANT, value=255
        )

        char = cv2.resize(char, (64, 64)) / 255.0
        char = char.reshape(1, 64, 64, 1)

        pred = cnn_model.predict(char, verbose=0)
        idx = str(np.argmax(pred))
        label = cnn_labels.get(idx)

        text.append(KANNADA_MAPPING.get(label, "?"))

    return " ".join(text)

# ================= GEMINI (FIXED) =================
def get_gemini_analysis(text):
    client = genai.Client(api_key=GOOG_API_KEY)

    prompt = f"""
You are an expert epigraphist.

Analyze the Halmidi inscription text:
"{text}"

Respond in THREE short sections:
1. Meaning (2–3 lines)
2. Historical significance (3–4 lines)
3. Linguistic note (2–3 lines)

Do not use markdown.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text.strip()

# ================= PUBLIC API =================
def analyze_image(image_path):
    text = perform_ocr(image_path)

    if not text:
        return {
            "ocr_text": "No characters detected",
            "analysis": ""
        }

    analysis = get_gemini_analysis(text)

    return {
        "ocr_text": text,
        "analysis": analysis
    }
