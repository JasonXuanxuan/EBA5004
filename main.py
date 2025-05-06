from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = "models/latest.joblib"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")


@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "ML prediction API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data.get("features")
        if not features or not isinstance(features, list):
            return jsonify({"error": "Missing or invalid 'features' field"}), 400

        input_array = np.array(features).reshape(1, -1)

        prediction = model.predict(input_array)
        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)