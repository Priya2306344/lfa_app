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
# UPLOAD IMAGE
# -------------------------
@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Get uploaded image
        file = request.files['image']

        # Process image
        result = process_image(file)

        # Add patient details
        result["Patient ID"] = request.form.get("patient_id")
        result["Patient Name"] = request.form.get("name")
        result["Age"] = request.form.get("age")
        result["Gender"] = request.form.get("gender")

        return jsonify(result)

    except Exception as e:
        print("ERROR:", e)          # This prints the error in Render logs
        return jsonify({
            "error": str(e)
        }), 500


# -------------------------
# RUN APP
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)