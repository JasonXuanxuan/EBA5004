import os
import joblib
import numpy as np
from flask import request, jsonify

from src.data.make_dataset import preprocess_absa_excel, extract_aspect_and_label
from src.models.train_model import ChineseQASystem
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import pandas as pd

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

def load_model_and_tokenizer(model_dir):
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    return model, tokenizer


def predict_sentiment(text, aspect, model, tokenizer, max_length=128,return_logits=False):
    inputs = tokenizer(text, aspect, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        if return_logits:
            probs = softmax(logits, dim=-1)
            return probs.numpy().flatten().tolist()
        else:
            return torch.argmax(logits, dim=1).item()

bert_model, tokenizer = load_model_and_tokenizer("models/bert_absa")
def predict_overallSentiment():
    file = request.files['file']
    try:
        df = preprocess_absa_excel(file)
    except Exception as e:
        return jsonify({"error": f"预处理失败: {str(e)}"}), 400

    result = []
    for _, row in df.iterrows():
        sentiment = predict_sentiment(row["text"], row["aspect"], bert_model, tokenizer)
        result.append({
            "text": row["text"],
            "aspect": row["aspect"],
            "predicted_label": sentiment
        })

    return jsonify(result)

def predict_singleSentiment():
    product_id = request.args.get("product_id")
    all_df = pd.read_excel("data/raw/all_comments.xlsx")
    if "商品ID" in all_df.columns:
        all_df = all_df.rename(columns={"商品ID": "product_id"})
    if "内容" in all_df.columns:
        all_df = all_df.rename(columns={"内容": "text"})

    all_df["product_id"] = all_df["product_id"].astype(str)
    product_id = str(product_id)
    df = all_df[all_df["product_id"] == product_id]

    if df.empty:
        return jsonify({"error": f"Not found record of product_id = {product_id}"}), 404

    df = df.rename(columns={"内容": "text"})
    df[["aspect", "label"]] = df["text"].apply(extract_aspect_and_label)

    absa_result = []
    for _, row in df.iterrows():
        sentiment = predict_sentiment(row['text'], row['aspect'], bert_model, tokenizer)
        absa_result.append({
            "text": row['text'],
            "aspect": row['aspect'],
            "predicted_label": sentiment
        })

    return jsonify({"product_id": product_id, "absa": absa_result})

def predict_sentimentScore():
    data = request.json
    text = data.get("text")

    if not text:
        return jsonify({"error": "缺少字段 'text'"}), 400

    aspect, _ = extract_aspect_and_label(text)

    probs = predict_sentiment(text, aspect, bert_model, tokenizer, return_logits=True)
    label_idx = probs.index(max(probs))
    label = ["negative", "neutral", "positive"][label_idx]

    return jsonify({
        "text": text,
        "aspect": aspect,
        "sentiment": label,
        "probabilities": {
            "negative": round(probs[0], 3),
            "neutral": round(probs[1], 3),
            "positive": round(probs[2], 3)
        }
    })