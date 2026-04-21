from flask import Flask, request, jsonify, redirect, render_template_string
import cv2
import numpy as np
import os
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# IMAGE PREPROCESSING (same idea as Pi)
# -----------------------------
def enhance_image(image):
    alpha = 1.8
    beta = 60
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# -----------------------------
# LINE DETECTION (FROM RASPBERRY PI)
# -----------------------------
def detect_lines(window):

    red_channel = window[:, :, 2].astype(float)

    if red_channel.max() == 0:
        return [], []

    red_norm = red_channel / red_channel.max()

    profile = red_norm.mean(axis=1)
    profile = 1 - profile

    profile = cv2.GaussianBlur(
        profile.reshape(-1, 1),
        (31, 1),
        0
    ).flatten()

    peaks = []
    intensities = []

    threshold = 0.08
    min_distance = 80

    for i in range(1, len(profile) - 1):

        if profile[i] > profile[i - 1] and profile[i] > profile[i + 1]:

            if profile[i] > threshold:

                if len(peaks) == 0 or i - peaks[-1] > min_distance:

                    peaks.append(i)

                    band_threshold = profile[i] * 0.5

                    top = i
                    bottom = i

                    while top > 0 and profile[top] > band_threshold:
                        top -= 1

                    while bottom < len(profile) - 1 and profile[bottom] > band_threshold:
                        bottom += 1

                    band_region = red_channel[top:bottom, :]

                    intensity = float(np.mean(band_region))
                    intensities.append(intensity)

    return peaks[:3], intensities[:3]

# -----------------------------
# SEVERITY LOGIC
# -----------------------------
def get_severity(intensities):

    if len(intensities) < 2:
        return "Invalid"

    test_intensity = intensities[1]

    if test_intensity < 200:
        return "Normal"
    elif test_intensity < 230:
        return "Moderate"
    else:
        return "High"

# -----------------------------
# LOAD IMAGE FROM REQUEST
# -----------------------------
def process_image(filepath):

    image = cv2.imread(filepath)

    if image is None:
        return [], []

    image = enhance_image(image)

    # NOTE: using full image as window (you can later improve cropping)
    return detect_lines(image)

# -----------------------------
# HOME ROUTE
# -----------------------------
@app.route("/")
def home():
    return redirect("/upload")

# -----------------------------
# UPLOAD PAGE + PROCESSING
# -----------------------------
@app.route("/upload", methods=["GET", "POST"])
def upload():

    if request.method == "GET":
        return render_template_string("""
        <h2>🧪 LFA Detection System</h2>

        <form method="POST" enctype="multipart/form-data">

            <label>Patient Name:</label><br>
            <input type="text" name="patient_name" required><br><br>

            <label>Age:</label><br>
            <input type="number" name="age" required><br><br>

            <label>Patient ID:</label><br>
            <input type="text" name="patient_id" value="{{id}}" readonly><br><br>

            <label>Capture / Upload Image:</label><br>
            <input type="file" name="image" accept="image/*" capture="camera" required><br><br>

            <button type="submit">Analyze</button>

        </form>
        """, id=str(uuid.uuid4())[:8])

    # ---------------- POST ----------------
    file = request.files["image"]

    patient_name = request.form.get("patient_name")
    age = request.form.get("age")
    patient_id = request.form.get("patient_id")

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    peaks, intensities = process_image(filepath)

    severity = get_severity(intensities)

    return jsonify({
        "patient_details": {
            "name": patient_name,
            "age": age,
            "patient_id": patient_id
        },
        "analysis": {
            "detected_lines": len(peaks),
            "intensities": intensities,
            "severity": severity
        }
    })

# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)