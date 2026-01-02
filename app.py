from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and scaler ONCE
model = joblib.load(os.path.join(BASE_DIR, "model", "knn_lung_cancer.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "model", "scaler.pkl"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        features = [
            int(request.form["gender"]),
            int(request.form["age"]),
            int(request.form["smoking"]),
            int(request.form["yellow_fingers"]),
            int(request.form["anxiety"]),
            int(request.form["peer_pressure"]),
            int(request.form["chronic_disease"]),
            int(request.form["fatigue"]),
            int(request.form["allergy"]),
            int(request.form["wheezing"]),
            int(request.form["alcohol"]),
            int(request.form["coughing"]),
            int(request.form["breath"]),
            int(request.form["swallowing"]),
            int(request.form["chest_pain"])
        ]

        features = np.array([features])
        features = scaler.transform(features)

        result = model.predict(features)
        prediction = "LUNG CANCER DETECTED" if result[0] == 1 else "NO LUNG CANCER"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    # Render requires host=0.0.0.0 and PORT from env
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
