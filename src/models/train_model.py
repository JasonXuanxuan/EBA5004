import pandas as pd
from sklearn.linear_model import LinearRegression
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
import logging
import os

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


def train_bert_absa_classifier(train_df, model_name='bert-base-chinese', num_labels=3, max_length=128, output_dir='models'):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    def tokenize(example):
        return tokenizer(example["text"], example["aspect"], truncation=True, padding="max_length", max_length=max_length)

    dataset = Dataset.from_pandas(train_df)
    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_test = dataset.train_test_split(test_size=0.2)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(filename="logs/training_output.log", level=logging.INFO, format="%(asctime)s - %(message)s")
    print(">>> TrainingArguments module:", TrainingArguments.__module__)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir='logs',
            logging_strategy="epoch",
            logging_steps=10,
            save_strategy="epoch",
            report_to="none"  # prevents WandB fallback errors
        ),
        train_dataset=train_test['train'],
        eval_dataset=train_test['test'],
    )

    logging.info("Starting training...")
    trainer.train()
    logging.info("Training completed.")
    eval_results = trainer.evaluate()
    logging.info(f"Evaluation Results: {eval_results}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
