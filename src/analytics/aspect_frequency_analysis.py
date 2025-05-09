import jieba.posseg as pseg
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
import platform
import matplotlib.pyplot as plt
from datetime import datetime
import spacy
from snownlp import SnowNLP
from tqdm import tqdm
import jieba

# Preprocess data for sentiment analysis
def clean_text(text):
    emoji_pattern = re.compile("[" 
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    text = re.sub(r'\s+', '', text)
    text = text.replace(",", "，").replace("!", "！").replace("?", "？").replace(".", "。")
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9。，！？]', '', text)
    return text

def set_plot_font():
    if platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'SimHei'
    elif platform.system() == 'Darwin':
        plt.rcParams['font.family'] = 'Arial Unicode MS'
    else:
        plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

def translate_labels(labels):
    translations = {
        "颜色": "Color","面料": "Fabric","质量": "Quality",
        "尺寸": "Size","尺码": "Fit","款式": "Style",
        "版型": "Cut","建议": "Recommendation","衣服": "Clothing",
        "裤子": "Pants","毛衣": "Sweater","秋季节": "Autumn Season",
        "物流": "Logistics","速度": "Speed","线头": "Loose Threads",
        "问题": "Issue","气质": "Temperament","客服": "Customer Service",
        "价格": "Price","效果": "Effect","舒适度": "Comfort",
        "穿搭": "Outfit Matching","性价比": "Cost Performance","购买": "Purchase",
        "色差": "Color Difference","感觉": "Feeling","做工": "Craftsmanship",
        "包装": "Packaging"
    }
    return [translations.get(label, label) for label in labels]

def visualize_aspects(counter, title="Aspect Frequency", top_n=10):
    if not counter:
        print("No aspect extracted.")
        return
    aspects, counts = zip(*counter.most_common(top_n))
    translated_aspects = translate_labels(aspects)
    plt.figure(figsize=(10, 6))
    plt.barh(translated_aspects[::-1], counts[::-1], color='skyblue')
    plt.xlabel("Frequency", fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(f"logs/aspect_frequency_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

def extract_nouns(text):
    words = pseg.cut(text)
    return [word for word, flag in words if flag.startswith('n') and len(word) > 1]

# === Extraction Methods ===
def extract_aspects_pos(df, col='cleaned_content'):
    print("[POS] Extracting nouns...")
    df['aspects_pos'] = df[col].apply(extract_nouns)
    counter = Counter()
    df['aspects_pos'].apply(lambda x: counter.update(x))
    visualize_aspects(counter, title="POS-based Aspect Frequency")
    return df, counter

def extract_aspects_dependency(df, col='cleaned_content'):
    print("[Dependency] Extracting using syntax...")
    nlp = spacy.load("zh_core_web_sm")
    all_aspects = []
    def get_aspects(doc):
        aspects = [token.text for token in doc
                   if token.pos_ == "NOUN" and token.dep_ in ["dobj", "pobj"] and len(token.text) > 1]
        all_aspects.extend(aspects)
        return aspects
    df['aspects_dep'] = [get_aspects(doc) for doc in nlp.pipe(df[col].tolist(), disable=["ner"])]
    counter = Counter(all_aspects)
    visualize_aspects(counter, title="Dependency-based Aspect Frequency")
    return df, counter

def extract_aspects_tfidf(df, col='cleaned_content', top_n=20):
    df['noun_tokens'] = df[col].apply(extract_nouns)
    all_nouns = [noun for tokens in df['noun_tokens'] for noun in tokens]
    noun_counter = Counter(all_nouns)

    def identity_tokenizer(text): return text

    vectorizer = TfidfVectorizer(tokenizer=identity_tokenizer, preprocessor=lambda x: x, token_pattern=None)
    tfidf_matrix = vectorizer.fit_transform(df['noun_tokens'])
    tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    tfidf_vocab = vectorizer.get_feature_names_out()
    tfidf_dict = dict(zip(tfidf_vocab, tfidf_scores))

    combined_score = {
        word: tfidf_dict.get(word, 0) * (1 + np.log(1 + freq))
        for word, freq in noun_counter.items()
    }

    sorted_aspects = sorted(combined_score.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_words = [w for w, _ in sorted_aspects]
    df['aspects_tfidf'] = df['noun_tokens'].apply(lambda x: [w for w in x if w in top_words])

    counter = Counter(dict(sorted_aspects))
    visualize_aspects(counter, title="TF-IDF-based Aspect Frequency")
    return df, counter

# === Main Entry Function ===
def run_all_aspect_extraction(file_path):
    set_plot_font()
    df = pd.read_excel(file_path, dtype={"商品ID": str})
    df = df[df['内容'] != '用户暂无评论'].copy()
    df.dropna(subset=["内容"], inplace=True)
    df['cleaned_content'] = df['内容'].astype(str).apply(clean_text)

    print("\n========== [1] POS-based Extraction ==========")
    df, counter_pos = extract_aspects_pos(df)

    print("\n========== [2] Dependency-based Extraction ==========")
    df, counter_dep = extract_aspects_dependency(df)

    print("\n========== [3] TF-IDF-based Extraction ==========")
    df, counter_tfidf = extract_aspects_tfidf(df)

    return {
        "df": df,
        "pos": counter_pos,
        "dep": counter_dep,
        "tfidf": counter_tfidf
    }

# ========== Utility Function ==========
def score_to_polarity(score):
    if score is None:
        return 'neutral'
    elif score >= 0.6:
        return 'positive'
    elif score <= 0.4:
        return 'negative'
    else:
        return 'neutral'

# ========== Aspect-Context Window Extraction ==========
def extract_aspect_contexts(text, aspects, window_size=6):
    """
    Given a text and its aspect terms, return context windows around each aspect using a fixed token window.
    """
    tokens = list(jieba.cut(text))
    contexts = []
    for i, token in enumerate(tokens):
        if token in aspects:
            start = max(0, i - window_size)
            end = min(len(tokens), i + window_size + 1)
            sub_text = ''.join(tokens[start:end])
            contexts.append((token, sub_text))
    return contexts

# ========== Core Function: Score Each Aspect with Context ==========
def aspect_context_sentiment_analysis(df, aspect_col='aspects', text_col='cleaned_content'):
    all_data = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scoring each aspect..."):
        text = row[text_col]
        aspects = row[aspect_col]
        aspect_contexts = extract_aspect_contexts(text, aspects)

        for aspect, context in aspect_contexts:
            score = SnowNLP(context).sentiments
            all_data.append({
                'index': idx,
                'original_text': row['内容'],
                'product_id': row['商品ID'] if '商品ID' in row else None,
                'aspect': aspect,
                'context': context,
                'score': score,
                'label': score_to_polarity(score)
            })

    aspect_df = pd.DataFrame(all_data)
    return aspect_df

def run_aspect_context_sentiment(df, aspect_col='aspects', text_col='cleaned_content'):
    """
    Score each extracted aspect in the given DataFrame using contextual sentiment.
    """
    if aspect_col not in df.columns:
        raise ValueError(f"Column '{aspect_col}' not found. Please run aspect extraction first.")

    return aspect_context_sentiment_analysis(df, aspect_col=aspect_col, text_col=text_col)