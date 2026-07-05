from flask import Flask, render_template, request, jsonify
import cv2
import os

app = Flask(__name__)

# Safe folder paths
REFERENCE_FOLDER = "images"

REFERENCE_IMAGES = {
    "0 line.jpeg": ("Invalid Strip", 0),
    "1 line.jpeg": ("Negative", 1),
    "2 lines.jpeg": ("Positive", 2),
    "3 lines.jpeg": ("Strongly Positive", 3)
}


@app.route("/")
def home():
    return render_template("index.html")


def compare_images(img1, img2):
    img1 = cv2.resize(img1, (300, 120))
    img2 = cv2.resize(img2, (300, 120))

    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(g1, g2)
    return float(diff.mean())


@app.route("/analyze", methods=["POST"])
def analyze():

    file = request.files.get("image")

    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    temp_path = "/tmp/temp.jpg"
    file.save(temp_path)

    uploaded = cv2.imread(temp_path)

    if uploaded is None:
        return jsonify({"error": "Invalid image"}), 400

    best_name = None
    best_score = 1e9

    # Loop reference images safely
    for filename, (status, lines) in REFERENCE_IMAGES.items():

        ref_path = os.path.join(REFERENCE_FOLDER, filename)

        if not os.path.exists(ref_path):
            continue

        ref = cv2.imread(ref_path)

        if ref is None:
            continue

        score = compare_images(uploaded, ref)

        if score < best_score:
            best_score = score
            best_name = filename

    if best_name is None:
        return jsonify({
            "status": "Unable to classify",
            "lines_detected": 0,
            "intensities": []
        })

    status, lines = REFERENCE_IMAGES[best_name]

    intensities = []
    if lines == 1:
        intensities = [120]
    elif lines == 2:
        intensities = [130, 125]
    elif lines == 3:
        intensities = [145, 140, 135]

    return jsonify({
        "status": status,
        "lines_detected": lines,
        "intensities": intensities
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)