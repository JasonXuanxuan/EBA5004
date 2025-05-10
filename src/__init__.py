from src.analytics.aspect_frequency_analysis import run_all_aspect_extraction, run_aspect_context_sentiment
from src.data.make_dataset import load_datasets, add_sentiment_scores, aggregate_sentiment
from src.features.build_features import merge_features, build_final_features, log_selected_features
from src.models.train_model import train_models, evaluate_model
from sklearn.model_selection import train_test_split
import joblib
import os
import json
from datetime import datetime
from pathlib import Path
from src.data.make_dataset import preprocess_absa_excel
from src.models.train_model import train_bert_absa_classifier

BASE_DIR = Path(__file__).resolve().parent.parent
def regression():
    comment_df, product_df = load_datasets(
        "data/raw/comment_dataset.xlsx",
        "data/raw/commodity_dataset.xlsx"
    )
    comment_df = add_sentiment_scores(comment_df)
    sentiment_summary = aggregate_sentiment(comment_df)
    merged = merge_features(product_df, sentiment_summary)

    X, y, scaler = build_final_features(merged)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    all_results = {}
    best_models = {}
    best_scores = {}

    for i in range(3):
        print(f"\n========== Training Round {i+1} ==========")
        models = train_models(X_train, y_train)
        run_results = {}
        for name, model in models.items():
            metrics = evaluate_model(model, X_test, y_test)
            print(f"{name} model -> Predicted sales metrics (Run {i+1}):", metrics)
            joblib.dump(model, f"models/{name}_model_run{i+1}.pkl")
            run_results[f"{name}_run{i+1}"] = metrics

            rmse = metrics["RMSE"]
            if name not in best_scores or rmse < best_scores[name]:
                best_models[name] = model
                best_scores[name] = rmse

        all_results[f"run_{i+1}"] = run_results

    for name, model in best_models.items():
        joblib.dump(model, f"models/{name}_model.pkl")

    joblib.dump(scaler, "models/scaler.pkl")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/model_metrics_{timestamp}.json"
    with open(log_path, "w") as log_file:
        json.dump(all_results, log_file, indent=4)

global results
def aspect_frequency():
    results = run_all_aspect_extraction("data/raw/comment_dataset.xlsx")
    df = results['df']
    aspect_level_df_tfidf = run_aspect_context_sentiment(df, aspect_col='aspects_tfidf', text_col='cleaned_content')
    aspect_level_df_pos = run_aspect_context_sentiment(df, aspect_col='aspects_pos', text_col='cleaned_content')
    aspect_level_df_dep = run_aspect_context_sentiment(df, aspect_col='aspects_dep', text_col='cleaned_content')
    aspect_level_df_tfidf.to_excel("logs/aspect_level_df_tfidf.xlsx", index=False)
    aspect_level_df_pos.to_excel("logs/aspect_level_df_pos.xlsx", index=False)
    aspect_level_df_dep.to_excel("logs/aspect_level_df_dep.xlsx", index=False)

def absa_model():
    df = preprocess_absa_excel("data/raw/comment_dataset.xlsx")
    train_bert_absa_classifier(
        train_df=df,
        model_name='bert-base-chinese',
        num_labels=3,
        max_length=128,
        output_dir='models/bert_absa'
    )