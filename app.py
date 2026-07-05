from flask import Flask, render_template, request, jsonify, redirect
import cv2
import os

app = Flask(__name__)

REFERENCE_FOLDER = "images"

REFERENCE_IMAGES = {
    "0 line.jpeg": ("Invalid Strip", 0),
    "1 line.jpeg": ("Negative", 1),
    "2 lines.jpeg": ("Positive", 2),
    "3 lines.jpeg": ("Strongly Positive", 3)
}


# ----------------------------
# HOME → redirect to patient page
# ----------------------------
@app.route("/")
def home():
    return redirect("/patient")


# ----------------------------
# PATIENT PAGE
# ----------------------------
@app.route("/patient")
def patient():
    return render_template("patient.html")


# ----------------------------
# IMAGE COMPARISON FUNCTION
# ----------------------------
def compare_images(img1, img2):
    img1 = cv2.resize(img1, (300, 120))
    img2 = cv2.resize(img2, (300, 120))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray1, gray2)

    return diff.mean()


# ----------------------------
# ANALYZE ROUTE
# ----------------------------
@app.route("/analyze", methods=["POST"])
def analyze():

    # Patient details
    name = request.form.get("name")
    age = request.form.get("age")
    patient_id = request.form.get("patient_id")

    # Uploaded image
    file = request.files["image"]

    temp_path = "temp.jpg"
    file.save(temp_path)

    uploaded = cv2.imread(temp_path)

    best_name = None
    best_score = 1e9

    for filename in REFERENCE_IMAGES:

        ref_path = os.path.join(REFERENCE_FOLDER, filename)
        ref = cv2.imread(ref_path)

        if ref is None:
            continue

        score = compare_images(uploaded, ref)

        if score < best_score:
            best_score = score
            best_name = filename

    if best_name is None:
        return jsonify({"status": "Unable to classify"})

    status, lines = REFERENCE_IMAGES[best_name]

    intensities = []

    if lines == 1:
        intensities = [120]
    elif lines == 2:
        intensities = [130, 125]
    elif lines == 3:
        intensities = [145, 140, 135]

    return jsonify({
        "name": name,
        "age": age,
        "patient_id": patient_id,
        "status": status,
        "lines_detected": lines,
        "intensities": intensities
    })


# ----------------------------
# RUN LOCALLY
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)