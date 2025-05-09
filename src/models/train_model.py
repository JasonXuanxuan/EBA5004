import pandas as pd
from sklearn.linear_model import LinearRegression
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb

class ChineseQASystem:
    def __init__(self, data_path="data/raw/comment_dataset.xlsx"):
        self.df = pd.read_excel(data_path)
        self.corpus = self.df["内容"].dropna().astype(str).tolist()

        # Build TF-IDF Vector
        self.vectorizer = TfidfVectorizer().fit(self.corpus)
        self.corpus_vectors = self.vectorizer.transform(self.corpus)

        # Loading model
        self.qa_pipeline = pipeline(
            task="question-answering",
            model="uer/roberta-base-chinese-extractive-qa",
            framework = "pt"
        )

    def answer(self, question: str) -> str:
        # Find the content segment most relevant to the question
        question_vector = self.vectorizer.transform([question])
        similarities = cosine_similarity(question_vector, self.corpus_vectors)
        best_idx = similarities.argmax()
        best_context = self.corpus[best_idx]

        # Extract answers
        result = self.qa_pipeline({
            "question": question,
            "context": best_context
        })
        return result["answer"]

class CustomRandomForest:
    def __init__(self, **kwargs):
        self.model = RandomForestRegressor(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

class CustomMLP:
    def __init__(self, **kwargs):
        self.model = MLPRegressor(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

class CustomLightGBM:
    def __init__(self, **kwargs):
        self.model = lgb.LGBMRegressor(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

def train_models(X_train, y_train):
    models = {
        "linear": LinearRegression(),
        "rf": CustomRandomForest(n_estimators=100),
        "mlp": CustomMLP(hidden_layer_sizes=(64, 32), max_iter=300),
        "lgb": CustomLightGBM()
    }
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}
