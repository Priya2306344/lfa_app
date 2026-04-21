from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# -----------------------------
# LFA ANALYSIS FUNCTION
# -----------------------------
def analyze_lfa(image_path):

    img = cv2.imread(image_path, 0)

    if img is None:
        return {
            "control_line": "Not Detected",
            "test_line": "Not Detected",
            "severity": "Invalid Image"
        }

    # Preprocessing (important for accuracy)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)

    h, w = img.shape

    # Split into regions (ASSUMPTION: top = control, bottom = test)
    control_region = thresh[0:h//2, :]
    test_region = thresh[h//2:h, :]

    control_score = np.sum(control_region == 255)
    test_score = np.sum(test_region == 255)

    # -----------------------------
    # LINE DETECTION
    # -----------------------------
    control_line = "Detected" if control_score > 5000 else "Not Detected"
    test_line = "Detected" if test_score > 5000 else "Not Detected"

    # -----------------------------
    # SEVERITY LOGIC
    # -----------------------------
    if test_line == "Not Detected":
        severity = "Normal"
    elif test_score < 15000:
        severity = "Moderate"
    else:
        severity = "High"

    return {
        "control_line": control_line,
        "test_line": test_line,
        "severity": severity
    }


# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    file = request.files["image"]
    patient_name = request.form.get("patient_name")

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    result = analyze_lfa(filepath)

    return jsonify({
        "patient_name": patient_name,
        "control_line": result["control_line"],
        "test_line": result["test_line"],
        "severity": result["severity"]
    })


# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)