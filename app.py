from flask import Flask, render_template, request, jsonify
from detect_strip import process_image
import traceback

app = Flask(__name__)


# -------------------------
# HOME PAGE
# -------------------------
@app.route('/')
def home():
    return render_template("index.html")


# -------------------------
# UPLOAD IMAGE
# -------------------------
@app.route('/upload', methods=['POST'])
def upload():

    try:

        print("========== NEW REQUEST ==========")
        print("FILES:", request.files)
        print("FORM:", request.form)

        # Check image exists
        if "image" not in request.files:
            return jsonify({
                "error": "No image received"
            }), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({
                "error": "No file selected"
            }), 400

        print("Image Name:", file.filename)

        # Process image
        result = process_image(file)

        # Add patient details
        result["Patient ID"] = request.form.get("patient_id")
        result["Patient Name"] = request.form.get("name")
        result["Age"] = request.form.get("age")
        result["Gender"] = request.form.get("gender")

        return jsonify(result)

    except Exception as e:

        print("========== ERROR ==========")
        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# -------------------------
# HEALTH CHECK
# -------------------------
@app.route('/test')
def test():
    return jsonify({
        "status": "Server is working"
    })


# -------------------------
# RUN APP
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)