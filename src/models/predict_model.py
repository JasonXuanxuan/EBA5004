import os
import joblib
import numpy as np
from flask import request, jsonify
from src.models.train_model import ChineseQASystem
import torch
from transformers import BertTokenizer, BertForSequenceClassification

qa_model = ChineseQASystem()
def qa():
    question = request.args.get("question")
    if not question:
        return jsonify({"error": "Missing 'question' in request"}), 400

    try:
        answer = qa_model.answer(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

MODEL_DIR = "models"
latest_feat_file = sorted(
    [f for f in os.listdir("logs") if f.startswith("selected_feature_names")],
    reverse=True
)[0]
with open(os.path.join("logs", latest_feat_file), "r") as f:
    selected_features = [line.strip() for line in f.readlines()]
def predict_sales():
    data = request.json
    model_name = data.get("model_name")
    model_path = os.path.join(MODEL_DIR, f"{model_name}_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    if not os.path.exists(model_path):
        return jsonify({"error": f"Model '{model_name}' not found."}), 404

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    except Exception as e:
        return jsonify({"error": f"Failed to load model or scaler: {str(e)}"}), 500

    try:
        X = np.array([[data.get(f, 0) for f in selected_features]])
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 400

    return jsonify({
        "model": model_name,
        "selected_features": selected_features,
        "input": data,
        "predicted_sales": float(prediction[0])
    })

def predict_overallSentiment():
    return jsonify({})
def predict_singleSentiment():
    return jsonify({})

def predict_sentimentScore():
    return jsonify({})