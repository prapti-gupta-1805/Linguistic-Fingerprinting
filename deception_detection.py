# =========================================
# LINGUISTIC FINGERPRINTING PIPELINE
# =========================================

import pandas as pd
import numpy as np
import re
import string
import nltk
import shap
import torch
import matplotlib.pyplot as plt

from scipy.sparse import hstack

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset

nltk.download('punkt')

RANDOM_STATE = 42

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data.csv")

df = df.rename(columns={'text_': 'text'})
df = df[['text', 'label']].dropna()

df['label'] = df['label'].map({'CG': 1, 'OR': 0})
df = df.dropna(subset=['label'])

df['label'] = df['label'].astype(int)
df['text'] = df['text'].astype(str)

print(f"Dataset Loaded: {df.shape}")

# =========================
# LINGUISTIC FEATURES
# =========================
class LinguisticFeatures(BaseEstimator, TransformerMixin):

    def get_features(self, text):
        words = nltk.word_tokenize(text)
        sentences = re.split(r'[.!?]+', text)

        num_words = len(words)
        num_sentences = len([s for s in sentences if s.strip()])
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        lexical_diversity = len(set(words)) / len(words) if words else 0
        punct_count = sum(c in string.punctuation for c in text)
        upper_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0

        return [
            num_words,
            num_sentences,
            avg_word_len,
            lexical_diversity,
            punct_count,
            upper_ratio
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self.get_features(t) for t in X])

# =========================
# FEATURE ENGINEERING
# =========================
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words='english'
)

X_text = tfidf.fit_transform(df['text'])

ling = LinguisticFeatures().transform(df['text'])
ling_scaled = StandardScaler().fit_transform(ling)

X = hstack([X_text, ling_scaled])
y = df['label']

print("Features ready")

# =========================
# MODELS
# =========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric='logloss'
    )
}

# =========================
# METRICS FUNCTION
# =========================
def print_model_results(log, name, y_test, preds):
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds)

    log("\n" + "="*60)
    log(f"MODEL: {name}")
    log("="*60)
    log(f"Accuracy: {acc:.4f}")

    log("\nClass-wise Performance:")
    for label, label_name in zip([0,1], ["Genuine (OR)", "Fake (CG)"]):
        log(f"  {label_name}:")
        log(f"    Precision: {report[str(label)]['precision']:.4f}")
        log(f"    Recall:    {report[str(label)]['recall']:.4f}")
        log(f"    F1-Score:  {report[str(label)]['f1-score']:.4f}")

    log("\nConfusion Matrix:")
    log("        Pred OR   Pred CG")
    log(f"Actual OR   {cm[0][0]}         {cm[0][1]}")
    log(f"Actual CG   {cm[1][0]}         {cm[1][1]}")

# =========================
# SPLITS
# =========================
splits = [0.1, 0.2, 0.3, 0.4]

for TEST_SIZE in splits:

    TRAIN_SIZE = int((1 - TEST_SIZE) * 100)
    TEST_PERCENT = int(TEST_SIZE * 100)

    RESULT_FILE = f"results_{TRAIN_SIZE}_{TEST_PERCENT}.txt"
    open(RESULT_FILE, "w").close()

    def log(msg):
        print(msg)
        with open(RESULT_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"\nRUNNING FOR SPLIT: {TRAIN_SIZE}-{TEST_PERCENT}")

    # SPLIT
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, np.arange(len(df)),
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    train_texts = df['text'].iloc[train_idx]
    test_texts = df['text'].iloc[test_idx]
    train_labels = y.iloc[train_idx]
    test_labels = y.iloc[test_idx]

    results_summary = {}

    # =========================
    # CLASSICAL MODELS
    # =========================
    log("\nTraining Classical Models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print_model_results(log, name, y_test, preds)
        results_summary[name] = accuracy_score(y_test, preds)

    # =========================
    # CROSS VALIDATION
    # =========================
    log("\nCROSS VALIDATION (5-FOLD)")
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5)
        log(f"{name}: Mean={scores.mean():.4f}, Std={scores.std():.4f}")

    # =========================
    # SHAP
    # =========================
    log("\nRunning SHAP...")
    sample_size = min(200, X_test.shape[0])
    X_dense = X_test[:sample_size].toarray()

    explainer = shap.TreeExplainer(models["XGBoost"])
    shap_values = explainer.shap_values(X_dense)

    feature_names = tfidf.get_feature_names_out().tolist() + [
        "num_words", "num_sentences", "avg_word_len",
        "lexical_diversity", "punct_count", "upper_ratio"
    ]

    shap_importance = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(shap_importance)[-10:]

    log("\nTop Features:")
    for i in reversed(top_indices):
        log(f"{feature_names[i]}: {shap_importance[i]:.4f}")

    # =========================
    # BERT (ALL SPLITS)
    # =========================
    log("\nTraining DistilBERT...")

    train_dataset = Dataset.from_dict({
        "text": list(train_texts),
        "labels": list(train_labels)
    })

    test_dataset = Dataset.from_dict({
        "text": list(test_texts),
        "labels": list(test_labels)
    })

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize(example):
        return tokenizer(example["text"], truncation=True)

    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    train_dataset = train_dataset.remove_columns(["text"])
    test_dataset = test_dataset.remove_columns(["text"])

    train_dataset.set_format("torch")
    test_dataset.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    bert_model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    ).to(device)

    training_args = TrainingArguments(
        output_dir=f"./bert_output_{TRAIN_SIZE}_{TEST_PERCENT}",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        save_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=bert_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator
    )

    trainer.train()

    preds = trainer.predict(test_dataset)
    y_pred = np.argmax(preds.predictions, axis=1)

    bert_acc = accuracy_score(test_labels, y_pred)
    print_model_results(log, "DistilBERT", test_labels, y_pred)

    results_summary["DistilBERT"] = bert_acc

    # =========================
    # FINAL PLOT
    # =========================
    plot_models = list(results_summary.keys())
    plot_scores = list(results_summary.values())

    plt.figure()
    bars = plt.bar(plot_models, plot_scores)

    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.title(f"Model Comparison ({TRAIN_SIZE}-{TEST_PERCENT} Split)")

    min_score = min(plot_scores)
    max_score = max(plot_scores)
    plt.ylim(min_score - 0.02, max_score + 0.02)

    for bar, score in zip(bars, plot_scores):
        plt.text(bar.get_x() + bar.get_width()/2, score, f"{score:.3f}",
                ha='center', va='bottom')

    plt.xticks(rotation=30)

    plot_file = f"plot_{TRAIN_SIZE}_{TEST_PERCENT}.png"
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close()

    log(f"\nSaved plot: {plot_file}")