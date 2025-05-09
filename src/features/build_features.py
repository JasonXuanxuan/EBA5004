import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_regression

def merge_features(product_df, sentiment_df):
    """
    Merge product features with sentiment aggregation by product_id.
    """
    sentiment_df.columns = [
        "product_id", "sentiment_mean", "sentiment_std",
        "sentiment_max", "sentiment_min", "comment_count"
    ]
    merged = product_df.merge(sentiment_df, left_on="product_code", right_on="product_id", how="left")
    return merged

def log_selected_features(df, selected_columns):
    """
    Save the selected feature values (first 5 rows) and selected feature names to log files.
    Also generate a bar chart for selected feature scores.
    """
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save sample values of selected features
    value_log_path = f"logs/selected_feature_values_{timestamp}.json"
    values = df[selected_columns].head(5).to_dict(orient="records")
    with open(value_log_path, "w") as f:
        json.dump(values, f, indent=4)
    print(f"[log] Selected feature values saved to {value_log_path}")

    # Save selected feature names
    name_log_path = f"logs/selected_feature_names_{timestamp}.txt"
    with open(name_log_path, "w") as f:
        for col in selected_columns:
            f.write(f"{col}\n")
    print(f"[log] Selected feature names saved to {name_log_path}")

def build_final_features(df, use_pca=False, n_components=5):
    df = df.fillna(0)
    y = df["sales_volume"]

    # Drop identifier or irrelevant columns
    drop_cols = ["sales_volume", "product_title", "product_image", "product_code", "product_id"]
    X_all = df.drop(columns=drop_cols, errors="ignore")

    # Compute F-scores
    f_vals, _ = f_regression(X_all, y)
    feature_scores = pd.Series(f_vals, index=X_all.columns)

    # Remove top 15 most and bottom 15 least correlated features
    to_exclude = feature_scores.sort_values(ascending=False).head(15).index.tolist() + \
                 feature_scores.sort_values(ascending=True).head(15).index.tolist()
    X_filtered = X_all.drop(columns=to_exclude, errors="ignore")

    # Select top features and ensure sentiment features are included
    remaining_scores = f_regression(X_filtered, y)[0]
    remaining_series = pd.Series(remaining_scores, index=X_filtered.columns)
    sentiment_features = ["sentiment_mean", "sentiment_std", "sentiment_max", "sentiment_min", "comment_count"]
    sentiment_included = [feat for feat in sentiment_features if feat in X_filtered.columns]

    selected_candidates = remaining_series.sort_values(ascending=False).index.tolist()
    final_features = []
    for feat in selected_candidates:
        if feat in sentiment_included or len([f for f in final_features if f in sentiment_included]) < 1:
            final_features.append(feat)
        elif len(final_features) < 15:
            final_features.append(feat)
        if len(final_features) == 15:
            break

    log_selected_features(df, final_features)

    # Save bar plot of selected feature scores
    score_map = {feat: remaining_series[feat] for feat in final_features if feat in remaining_series}
    plt.figure(figsize=(12, 6))
    plt.bar(score_map.keys(), score_map.values())
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("F-score")
    plt.title("Selected Feature Importance (including sentiment)")
    plt.tight_layout()
    plt.savefig(f"logs/feature_importance_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    print("[log] Feature importance plot saved.")

    # Scale selected features
    X_selected = X_filtered[final_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    # Apply PCA if enabled
    if use_pca:
        pca = PCA(n_components=n_components)
        X_scaled = pca.fit_transform(X_scaled)

    return X_scaled, y, scaler

