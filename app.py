from flask import Flask, request, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def analyze_lfa(image_path):

    img = cv2.imread(image_path, 0)

    if img is None:
        return {
            "lines_detected": {
                "control": "Not Detected",
                "test": "Not Detected"
            },
            "intensity": 0,
            "severity": "Invalid Image"
        }

    # Preprocess
    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)

    h, w = img.shape

    # Split strip
    control_region = thresh[0:h//2, :]
    test_region = thresh[h//2:h, :]

    control_score = np.sum(control_region == 255)
    test_score = np.sum(test_region == 255)

    control_line = "Detected" if control_score > 5000 else "Not Detected"
    test_line = "Detected" if test_score > 5000 else "Not Detected"

    # Better intensity (use both scores)
    intensity = float((control_score + test_score) / 2)

    # Improved severity logic
    if test_score < 2000:
        severity = "Normal"
    elif test_score < 5000:
        severity = "Moderate"
    else:
        severity = "High"

    return {
        "lines_detected": {
            "control": control_line,
            "test": test_line
        },
        "intensity": intensity,
        "severity": severity
    }


@app.route("/")
def home():
    return "LFA App is Running"


@app.route("/upload", methods=["POST"])
def upload():

    file = request.files["image"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    result = analyze_lfa(filepath)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)