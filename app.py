from flask import Flask, render_template, request, jsonify
from detect_strip import process_image

app = Flask(__name__)

# -------------------------
# HOME PAGE
# -------------------------
@app.route('/')
def home():
    return render_template("index.html")


# -------------------------
# UPLOAD IMAGE AND ANALYZE
# -------------------------
@app.route('/upload', methods=['POST'])
def upload():

    # Get uploaded image
    file = request.files['image']

    # Process image
    result = process_image(file)

    # Get patient details
    patient_id = request.form.get("patient_id")
    name = request.form.get("name")
    age = request.form.get("age")
    gender = request.form.get("gender")

    # Add patient details to output
    result["Patient ID"] = patient_id
    result["Patient Name"] = name
    result["Age"] = age
    result["Gender"] = gender

    return jsonify(result)


# -------------------------
# RUN APP
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)