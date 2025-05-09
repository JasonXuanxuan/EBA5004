from flask import Flask, jsonify
from src.models.predict_model import qa, predict_sales, predict_overallSentiment, predict_singleSentiment, \
    predict_sentimentScore

app = Flask(__name__)

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"response": "pong"})

app.add_url_rule("/api/qa", view_func=qa)

app.add_url_rule("/api/salesPredict", view_func=predict_sales, methods=["POST"])

app.add_url_rule("/api/overallSentiment", view_func=predict_overallSentiment, methods=["POST"])

app.add_url_rule("/api/singleSentiment", view_func=predict_singleSentiment, methods=["GET"])

app.add_url_rule("/api/sentimentScore", view_func=predict_sentimentScore, methods=["GET"])

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)