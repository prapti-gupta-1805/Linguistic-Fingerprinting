# Linguistic Fingerprinting for Deception Detection

An NLP and machine learning project for detecting deceptive online reviews using hybrid feature engineering, classical machine learning, and transformer-based language models.

## Overview

This project explores whether deceptive reviews exhibit identifiable linguistic fingerprints that can be used for automated classification.

The system combines TF-IDF representations with handcrafted linguistic features such as lexical diversity, sentence structure, punctuation usage, and capitalization patterns. Multiple machine learning models were evaluated and compared against a fine-tuned DistilBERT baseline.

Experiments were conducted on a balanced dataset containing **40,432 online reviews**.

---

## Tech Stack

### Data Processing & NLP
- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn

### Machine Learning
- TF-IDF
- Logistic Regression
- Linear SVM
- XGBoost

### Deep Learning
- PyTorch
- DistilBERT
- Hugging Face Transformers
- Hugging Face Datasets

### Explainability & Visualization
- SHAP
- Matplotlib

---

## Methodology

### Data Preprocessing

- Missing value removal
- Label encoding
- Text normalization
- Tokenization using NLTK
- Stop-word removal

### Feature Engineering

#### TF-IDF Features

- Unigrams and bigrams
- Maximum feature size: 5000

#### Linguistic Fingerprinting Features

- Word count
- Sentence count
- Average word length
- Lexical diversity
- Punctuation frequency
- Uppercase character ratio

The linguistic features were normalized and combined with TF-IDF vectors to create a hybrid feature space.

---

## Models Evaluated

### Classical Machine Learning

- Logistic Regression
- Linear SVM
- XGBoost

### Transformer Baseline

- DistilBERT fine-tuned for binary text classification

---

## Results

| Model | Best Accuracy |
|---------|------------|
| Logistic Regression | 90.36% |
| Linear SVM | 90.95% |
| XGBoost | 89.93% |
| DistilBERT | 98.24% |

### Key Findings

- Linear SVM achieved the strongest performance among traditional machine learning models.
- DistilBERT achieved **98.24% accuracy**, **99.14% precision**, and **98.26% F1-score**.
- Transformer-based approaches outperformed classical machine learning models by more than **7 percentage points**.
- Linguistic fingerprinting features improved both classification performance and interpretability.
- SHAP analysis highlighted lexical diversity, review length, and stylistic patterns as influential indicators of deception.


## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python deception_detection.py
```

## License

This project is intended for educational and research purposes.