from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

def analyze_lfa(image_path):
    img = cv2.imread(image_path, 0)

    # Safety check (important)
    if img is None:
        return 0, "Image not found"

    intensity = np.mean(img)

    if intensity < 80:
        severity = "High"
    elif intensity < 150:
        severity = "Moderate"
    else:
        severity = "Low"

    return intensity, severity


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"})

    file = request.files['image']
    file_path = "image.png"
    file.save(file_path)

    intensity, severity = analyze_lfa(file_path)

    return jsonify({
        "intensity": float(intensity),
        "severity": severity
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')