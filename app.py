from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import traceback

# 🔴 CORRECT IMPORT (THIS FIXES THE ERROR)
from analyze_with_gemini import analyze_image

# ================= FLASK SETUP =================
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= TEST ROUTE =================
@app.route("/", methods=["GET"])
def home():
    return "Backend is running", 200

# ================= MAIN API =================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image received"}), 400

        image = request.files["image"]
        if image.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        image_path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(image_path)

        # 🔴 THIS LINE WAS FAILING BEFORE
        result = analyze_image(image_path)

        return jsonify({
            "kannada_text": result["ocr_text"],
            "analysis": result["analysis"]
        }), 200

    except Exception as e:
        traceback.print_exc()  # ALWAYS SHOW REAL ERROR
        return jsonify({"error": str(e)}), 500

# ================= RUN =================
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
