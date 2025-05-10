## ManFang Clothing Online (MFCO) — Aspect-Based Sentiment Analysis and Forecast System
This project presents a Natural Language Processing (NLP)-powered decision support system tailored for **ManFang Clothing Online (MFCO)**, a Douyin-based e-commerce clothing retailer. It is developed to tackle key business pain points using advanced machine learning and sentiment analysis.

---

### Project Objectives

ManFang currently faces several challenges:

- **Product selection difficulty**: Consumers struggle to identify the best items among vast options.
- **Stagnant sales growth**: Limited data-driven feedback to adjust product strategies.
- **Unquantified user opinions**: Comments remain underutilized in decision-making.
- **High customer support cost**: Manual service is expensive and inconsistent.

To address these, our system integrates:

- **Aspect-Based Sentiment Analysis (ABSA)** to extract opinions on specific product features (e.g., price, fit, fabric)
- **Comment-driven sales forecasting**
- **A QA-based extraction interface for customer inquiries**
- **Model training and deployment pipeline via Flask API**

---

### Project Structure

EBA5004/  
├── data/  
│ ├── raw/ # Original comment datasets (e.g., all_comments.xlsx)  
│ ├── processed/ # Cleaned and labeled data  
│ └── external/ # External product metadata  
│  
├── logs/ # Model training logs  
│  
├── models/ # Exported fine-tuned BERT models & tokenizers  
│  
├── src/  
│ ├── analytics/ # Frequency and keyword analysis scripts  
│ ├── data/ # Preprocessing and extraction (e.g., aspect, label)  
│ ├── features/ # Feature engineering (for forecasting models)  
│ ├── models/ # Model training & Flask API prediction logic  
│ ├── init.py  
│ └── main.py # App entry point — launches the web API  
│
├── main.py # Optional CLI launcher  
└── README.md # You are here.  

---

### Environment
No restrict request

---

### pip mandatory Packages
~~~ bash
pip install -r requirements.txt
~~~

### Train models
```bash
python src/main.py
```

### Run on 0000:5000
```bash
python main.py
```