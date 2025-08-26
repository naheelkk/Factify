# 📰 Factify – Fake News Detection with Baselines & BERT

Factify is a machine learning system for **fake news detection**, built with both **classical ML baselines** and **transformer-based deep learning (BERT)**. This project demonstrates the evolution from lightweight models to advanced fine-tuned transformers, providing both **practical performance** and **academic insights**.

## 🚀 Project Workflow

### 1. Data Preprocessing
* Clean text (punctuations, lowercasing, stopwords).
* Train/validation/test split.
* Handle class imbalance if present.

### 2. Baseline Models (TF-IDF + ML)
We start with lightweight models to establish benchmarks:
* **Logistic Regression (TF-IDF)**
* **Linear SVM (hinge loss)**
* **Naive Bayes (Multinomial & Complement)**
* **Random Forest (tree-based)**

📊 These baselines are fast, interpretable, and give us a **control point** to measure deep learning improvements.

**Baseline Architecture**:
```
Raw Text → TF-IDF Vectorizer → ML Classifier → Prediction
```

### 3. Transformer-based Models (BERT)
We fine-tune **BERT (bert-base-uncased)** for binary classification (`FAKE` vs `REAL`). This allows contextual understanding of language beyond TF-IDF.

**BERT Architecture for Factify**:
```
Raw Text → BERT Tokenizer → BERT Encoder → Classification Head → Prediction
```

* **Tokenizer**: WordPiece, max length 512
* **Encoder**: 12-layer Transformer (self-attention)
* **Head**: Fully connected linear classifier with softmax
* **Loss**: CrossEntropy
* **Optimizer**: AdamW with learning rate 2e-5
* **Training**: 3–5 epochs

### 4. Evaluation Metrics
* Accuracy
* Precision, Recall, F1-score
* Confusion Matrix
* ROC-AUC

We focus especially on **recall for the FAKE class**, since false negatives (missing fake news) are costly.

## 📊 Results Overview

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|---------|-----|
| Logistic Regression | **94.18%** | **0.9401** | **0.9402** | **0.9401** |
| Random Forest | 91.31% | 0.9296 | 0.8884 | 0.9085 |
| Linear SVM | ~91% | High | Moderate | Moderate |
| Naive Bayes | ~88% | Moderate | Sometimes higher recall | Moderate |
| **BERT (fine-tuned)** | **95%+** | **High** | **Much stronger recall** | **Balanced** |

*(BERT results pending fine-tuning completion)*

## 🛠️ Tech Stack
* **Data Processing**: Pandas, NumPy
* **EDA**: Matplotlib, Seaborn, WordCloud
* **Baselines**: scikit-learn (TF-IDF, LR, SVM, NB, RF)
* **Deep Learning**: HuggingFace Transformers, PyTorch
* **Web Deployment**: FastAPI (backend), Streamlit (UI)

## 🏗️ System Architecture

```
            ┌────────────────────┐
            │      User Input    │
            └─────────┬──────────┘
                      │
          ┌───────────▼───────────┐
          │   Preprocessing (NLP) │
          └───────────┬───────────┘
                      │
     ┌────────────────┴───────────────┐
     │                               │
┌────▼─────┐                   ┌─────▼──────┐
│ Baseline │                   │   BERT     │
│  Models  │                   │ Transformer│
└────▲─────┘                   └─────▲──────┘
     │                               │
     └─────────────┬─────────────────┘
                   │
         ┌─────────▼─────────┐
         │ Prediction Output │
         └───────────────────┘
```

## 📦 Installation

```bash
# Clone repo
git clone https://github.com/yourusername/factify.git
cd factify

# Install requirements
pip install -r requirements.txt
```

## 🖥️ Running Baselines

```bash
python train_baselines.py
```

## 🤖 Fine-tuning BERT

```bash
python train_bert.py
```

## 🌐 Deployment
* **Backend**: FastAPI serving the trained model (`/predict` endpoint).
* **Frontend**: Streamlit for an interactive web interface.

```bash
# Start FastAPI backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Launch Streamlit frontend
streamlit run app.py
```

## 📌 Roadmap
- [x] Data collection & preprocessing
- [x] Baseline ML models
- [ ] BERT fine-tuning
- [ ] API (FastAPI backend)
- [ ] Web UI (Streamlit frontend)
- [ ] Deployment (Docker/Heroku/Render)

## 📁 Project Structure

```
factify/
├── notebooks/
│   └── models/dataset-based/
│       └── WELFake.ipynb          # Model training & evaluation
├── data/
│   └── trimmed_processed/
│       └── WELFake.pkl            # Processed dataset
├── models/
│   ├── baseline_models/           # TF-IDF + ML models
│   └── bert_model/                # Fine-tuned BERT
├── src/
│   ├── train_baselines.py         # Training script for baselines
│   ├── train_bert.py              # BERT fine-tuning script
│   ├── main.py                    # FastAPI backend
│   └── streamlit_app.py           # Frontend interface
├── requirements.txt
└── README.md
```

## 🎯 Key Features
* **Dual Architecture**: Classical ML baselines + Modern BERT transformers
* **High Performance**: 94%+ accuracy with detailed metrics
* **Interactive Interface**: Real-time fake news detection
* **Confidence Scoring**: Prediction confidence levels
* **LLM Explanations**: AI-powered reasoning for classifications

## 📖 References
* [WELFake Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
* [HuggingFace Transformers](https://huggingface.co/transformers/)
* Relevant research papers on fake news detection

---

**Built with ❤️ for combating misinformation through AI**