import pandas as pd
from snownlp import SnowNLP
from tqdm import tqdm
import jieba

# 1.Preprocess data for regression
def load_datasets(comment_path, commodity_path):
    comment_df = pd.read_excel(comment_path, engine="openpyxl")
    commodity_df = pd.read_excel(commodity_path, engine="openpyxl")
    comment_df = rename_comment_columns(comment_df)
    commodity_df = rename_commodity_columns(commodity_df)
    return comment_df, commodity_df


def rename_comment_columns(df):
    return df.rename(columns={
        "商品ID": "product_id",
        "日期": "date",
        "评论人": "user",
        "内容": "content"
    })


def rename_commodity_columns(df):
    return df.rename(columns={
        "商品标题": "product_title",
        "商品图片": "product_image",
        "商品编号": "product_code",
        "成交金额": "transaction_amount",
        "成交订单数": "order_count",
        "成交件数": "sales_volume",
        "成交人数": "buyer_count",
        "预估佣金收入": "estimated_commission",
        "实际佣金收入": "actual_commission",
        "新用户成交人数": "new_buyer_count",
        "曝光人数": "exposure_users",
        "曝光次数": "exposure_times",
        "点击人数": "click_users",
        "点击次数": "click_times",
        "退款金额": "refund_amount",
        "退款订单数": "refund_orders",
        "退货件数": "refund_items",
        "退款人数": "refund_users",
        "退款率": "refund_rate",
        "品质退货率（滞后）": "quality_refund_rate_lag",
        "投诉率（滞后）": "complaint_rate_lag",
        "评价数": "review_count",
        "好评数": "positive_reviews",
        "差评数": "negative_reviews",
        "好评率": "positive_rate",
        "差评率": "negative_rate",
        "直播间商品曝光人数": "live_exposure_users",
        "直播间商品点击人数": "live_click_users",
        "直播间成交金额": "live_transaction_amount",
        "直播间成交订单数": "live_order_count",
        "直播间成交件数": "live_sales_volume",
        "直播间成交人数": "live_buyer_count",
        "直播间曝光-支付转化率": "live_exposure_to_payment_rate",
        "直播间点击-支付转化率": "live_click_to_payment_rate",
        "直播间曝光-点击转化率": "live_exposure_to_click_rate",
        "直播间退款金额": "live_refund_amount",
        "直播间退款人数": "live_refund_users",
        "直播间退款订单数": "live_refund_orders",
        "短视频商品曝光人数": "video_exposure_users",
        "短视频商品点击人数": "video_click_users",
        "短视频成交金额": "video_transaction_amount",
        "短视频成交订单数": "video_order_count",
        "短视频成交件数": "video_sales_volume",
        "短视频成交人数": "video_buyer_count",
        "短视频曝光-支付转化率": "video_exposure_to_payment_rate",
        "短视频点击-支付转化率": "video_click_to_payment_rate",
        "短视频曝光-点击转化率": "video_exposure_to_click_rate",
        "短视频退款金额": "video_refund_amount",
        "短视频退款人数": "video_refund_users",
        "短视频退款订单数": "video_refund_orders",
        "商品卡曝光人数": "card_exposure_users",
        "商品卡点击人数": "card_click_users",
        "商品卡成交金额": "card_transaction_amount",
        "商品卡成交订单数": "card_order_count",
        "商品卡成交件数": "card_sales_volume",
        "商品卡成交人数": "card_buyer_count",
        "商品卡曝光-支付转化率": "card_exposure_to_payment_rate",
        "商品卡点击-支付转化率": "card_click_to_payment_rate",
        "商品卡退款金额": "card_refund_amount",
        "商品卡退款人数": "card_refund_users",
        "商品卡退款订单数": "card_refund_orders"
    })


def add_sentiment_scores(comment_df):
    comment_df["sentiment_score"] = comment_df["content"].apply(lambda x: SnowNLP(str(x)).sentiments)
    return comment_df


def aggregate_sentiment(comment_df):
    return comment_df.groupby("product_id").agg({
        "sentiment_score": ["mean", "std", "max", "min", "count"]
    }).reset_index()

# # 2.Preprocess for sentiment analytics
# # ========== Utility Function ==========
# def score_to_polarity(score):
#     if score is None:
#         return 'neutral'
#     elif score >= 0.6:
#         return 'positive'
#     elif score <= 0.4:
#         return 'negative'
#     else:
#         return 'neutral'
#
#
# # ========== Aspect-Context Window Extraction ==========
# def extract_aspect_contexts(text, aspects, window_size=6):
#     """
#     Given a text and its aspect terms, return context windows around each aspect using a fixed token window.
#     """
#     tokens = list(jieba.cut(text))
#     contexts = []
#     for i, token in enumerate(tokens):
#         if token in aspects:
#             start = max(0, i - window_size)
#             end = min(len(tokens), i + window_size + 1)
#             sub_text = ''.join(tokens[start:end])
#             contexts.append((token, sub_text))
#     return contexts
#
#
# # ========== Core Function: Score Each Aspect with Context ==========
# def aspect_context_sentiment_analysis(df, aspect_col='aspects', text_col='cleaned_content'):
#     all_data = []
#     for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scoring each aspect..."):
#         text = row[text_col]
#         aspects = row[aspect_col]
#         aspect_contexts = extract_aspect_contexts(text, aspects)
#
#         for aspect, context in aspect_contexts:
#             score = SnowNLP(context).sentiments
#             all_data.append({
#                 'index': idx,
#                 'original_text': row['内容'],
#                 'product_id': row['商品ID'] if '商品ID' in row else None,
#                 'aspect': aspect,
#                 'context': context,
#                 'score': score,
#                 'label': score_to_polarity(score)
#             })
#
#     aspect_df = pd.DataFrame(all_data)
#     return aspect_df
#
#
# # ========== Main Entrypoint ==========
# def run_aspect_context_sentiment(df, aspect_col='aspects', text_col='cleaned_content'):
#     """
#     Score each extracted aspect in the given DataFrame using contextual sentiment.
#     """
#     if aspect_col not in df.columns:
#         raise ValueError(f"Column '{aspect_col}' not found. Please run aspect extraction first.")
#
#     return aspect_context_sentiment_analysis(df, aspect_col=aspect_col, text_col=text_col)
#
# # ========== Pseudo Labeling Function ==========
# def generate_pseudo_label(row, pos_words=None, neg_words=None):
#     """
#     Generate pseudo labels using aspect context + SnowNLP + sentiment word rules:
#     - 1: Positive (score >= 0.65 or positive word matched)
#     - 0: Negative (score <= 0.35 or negative word matched)
#     - 2: Neutral (otherwise)
#     """
#     text = row['cleaned_content']
#     aspect = row['aspect']
#
#     if pos_words is None:
#         pos_words = ["喜欢", "好看", "满意", "舒服", "推荐", "柔软", "合适", "贴身", "显瘦", "值", "划算"]
#     if neg_words is None:
#         neg_words = ["失望", "难看", "差", "瑕疵", "小", "紧", "大", "不值", "掉色", "粗糙", "不舒服"]
#
#     tokens = list(jieba.cut(text))
#     try:
#         idx = tokens.index(aspect)
#     except ValueError:
#         return 2  # Neutral if aspect not found
#
#     window = tokens[max(0, idx - 6):idx + 7]
#     context = ''.join(window)
#     score = SnowNLP(context).sentiments
#
#     if any(word in context for word in pos_words):
#         return 1
#     elif any(word in context for word in neg_words):
#         return 0
#     elif score >= 0.65:
#         return 1
#     elif score <= 0.35:
#         return 0
#     else:
#         return 2
#
# # ========== Main Export Function ==========
# def export_aspect_training_with_pseudo_label(
#     results,
#     method='tfidf',
#     text_col='cleaned_content',
#     output_path='aspect_training_pseudo.xlsx'
# ):
#     """
#     Generate training samples (text + aspect + pseudo label) from extracted aspects.
#     """
#     if method not in results or 'df' not in results:
#         raise ValueError("The 'results' must contain both 'df' and the specified method's aspect counter.")
#
#     df = results['df'].copy()
#     aspect_counter = results[method]
#
#     # Extract aspects from each review
#     df['aspects'] = df[text_col].apply(lambda text: [a for a in aspect_counter if a in text])
#
#     records = []
#     for _, row in df.iterrows():
#         for aspect in row['aspects']:
#             records.append({
#                 'product_id': row.get('商品ID', ''),
#                 'original_text': row['内容'],
#                 'cleaned_content': row[text_col],
#                 'aspect': aspect
#             })
#
#     aspect_df = pd.DataFrame(records)
#
#     # Apply pseudo label function (optional - uncomment to label)
#     aspect_df.to_excel(output_path, index=False)
#     # return aspect_df