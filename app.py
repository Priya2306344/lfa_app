from flask import Flask, render_template, request, jsonify

from detect_strip import process_image

app = Flask(__name__)


# -------------------------
# HOME
# -------------------------
@app.route('/')
def home():
    return render_template("index.html")


# -------------------------
# UPLOAD IMAGE
# -------------------------
@app.route('/upload', methods=['POST'])
def upload():

    file = request.files['image']

    result = process_image(file)

    return jsonify(result)


# -------------------------
# RUN APP
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)