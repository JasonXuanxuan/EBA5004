import pandas as pd
from snownlp import SnowNLP
import pandas as pd
from src.analytics.aspect_frequency_analysis import extract_nouns
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import torch
from collections import Counter

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
import jieba.posseg as pseg
def extract_aspect_and_label(text):
    words = pseg.cut(text)
    aspects = [word for word, flag in words if flag.startswith('n')]
    primary_aspect = aspects[0] if aspects else "通用"

    positive_keywords = ["好看", "舒服", "喜欢", "合适", "便宜", "满意", "完好", "质感", "无可挑剔"]
    negative_keywords = ["退", "差", "不好", "异味", "不行", "不喜欢", "问题", "不满意", "失望"]

    label = 1
    if any(word in text for word in positive_keywords):
        label = 2
    elif any(word in text for word in negative_keywords):
        label = 0

    return pd.Series([primary_aspect, label])


def preprocess_absa_excel(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path)

    if "内容" not in df.columns:
        raise ValueError("缺少评论内容列 '内容'")

    df = df.rename(columns={"内容": "text"})

    # 自动提取 aspect 和 label
    df[["aspect", "label"]] = df["text"].apply(extract_aspect_and_label)

    return df[["text", "aspect", "label"]]
