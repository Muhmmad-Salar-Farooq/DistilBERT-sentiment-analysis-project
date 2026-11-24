<div align="center">

# 🎬 Movie Review Sentiment Analysis with DistilBERT

**Fine-tuned DistilBERT that classifies movie reviews as Positive or Negative.**
</div>

---

## 📌 Overview

This project fine-tunes [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased) for **binary sentiment classification** on the [IMDb](https://huggingface.co/datasets/stanfordnlp/imdb) movie review dataset. DistilBERT is about 40% smaller and 60% faster than BERT, which makes it ideal for quick experiments on a free Colab GPU.

## ✨ Highlights

- 🧠 Transfer learning with a lightweight transformer
- ⚡ Strong results from a small subset of the data (4,000 training reviews)
- ⚙️ Reproducible setup with a single `Config` class
- 📊 Evaluated with accuracy, F1 score and a confusion matrix
- 🔮 Easy inference using the Hugging Face `pipeline`

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Model | DistilBERT (`distilbert-base-uncased`) |
| Framework | PyTorch, Hugging Face Transformers |
| Dataset | IMDb (25,000 train / 25,000 test reviews) |
| Metrics | scikit-learn |
| Environment | Google Colab (T4 GPU) |

## ⚙️ Training Configuration

| Parameter | Value |
|---|---|
| Training samples | 4,000 |
| Test samples | 1,000 |
| Max sequence length | 256 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Epochs | 2 |

## 📈 Results

| Metric | Score |
|---|---|
| Accuracy | `XX.X%` |
| F1 Score | `XX.X%` |

## 🚀 Quick Start

**1. Install dependencies**

```bash
pip install transformers datasets accelerate scikit-learn
```

**2. Run inference**

```python
from transformers import pipeline

clf = pipeline("sentiment-analysis", model="my-distilbert-sentiment")

print(clf("I loved every minute of this film, the acting was brilliant!"))
print(clf("What a waste of time. The plot made no sense."))
```

**Example output**

```
[{'label': 'POSITIVE', 'score': 0.99}]
[{'label': 'NEGATIVE', 'score': 0.98}]
```

## 🔬 How It Works

1. **Tokenize** reviews with the DistilBERT WordPiece tokenizer
2. **Add** a classification head on top of the `[CLS]` token representation
3. **Fine-tune** the model with the Hugging Face `Trainer`
4. **Evaluate** on unseen reviews and test on custom sentences

## 🗂️ Project Structure

```
├── sentiment_distilbert.ipynb    # Full training notebook
└── README.md
```

## 🔭 Future Work

- Train on the full 25,000 reviews for higher accuracy
- Compare with `bert-base-uncased` and `roberta-base`
- Extend to 3 classes (positive, neutral, negative)
- Build a live demo with Gradio or Streamlit

## 👤 Author

**Muhammad Salar Farooq**
[GitHub](https://github.com/Muhmmad-Salar-Farooq) · [LinkedIn](https://linkedin.com/in/muhammad-salar-farooq-b808113aa)

---

<div align="center">
⭐ If you found this project useful, please give it a star!
</div>
