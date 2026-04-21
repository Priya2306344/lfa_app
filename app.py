from flask import Flask, request, jsonify, redirect, render_template_string
import cv2
import numpy as np
import os
import uuid

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
            "lines_detected": {
                "control": "Not Detected",
                "test": "Not Detected"
            },
            "intensity": 0,
            "severity": "Invalid Image"
        }

    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)

    h, w = img.shape

    control_region = thresh[0:h//2, :]
    test_region = thresh[h//2:h, :]

    control_score = np.sum(control_region == 255)
    test_score = np.sum(test_region == 255)

    control_line = "Detected" if control_score > 5000 else "Not Detected"
    test_line = "Detected" if test_score > 5000 else "Not Detected"

    intensity = float((control_score + test_score) / 2)

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


# -----------------------------
# HOME ROUTE
# -----------------------------
@app.route("/")
def home():
    return redirect("/upload")


# -----------------------------
# UPLOAD + PATIENT FORM
# -----------------------------
@app.route("/upload", methods=["GET", "POST"])
def upload():

    if request.method == "GET":
        return render_template_string("""
        <h2>LFA Disease Detection System</h2>

        <form method="POST" enctype="multipart/form-data">

            <label>Patient Name:</label><br>
            <input type="text" name="patient_name" required><br><br>

            <label>Age:</label><br>
            <input type="number" name="age" required><br><br>

            <label>Patient ID:</label><br>
            <input type="text" name="patient_id" value="{{id}}" readonly><br><br>

            <label>Upload LFA Image:</label><br>
            <input type="file" name="image" accept="image/*" capture="camera" required><br><br>

            <button type="submit">Analyze</button>

        </form>
        """, id=str(uuid.uuid4())[:8])

    # -----------------------------
    # POST METHOD
    # -----------------------------
    file = request.files["image"]

    patient_name = request.form.get("patient_name")
    age = request.form.get("age")
    patient_id = request.form.get("patient_id")

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    result = analyze_lfa(filepath)

    return jsonify({
        "patient_details": {
            "name": patient_name,
            "age": age,
            "patient_id": patient_id
        },
        "result": result
    })


# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)