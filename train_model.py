"""
train_model.py - Machine Learning Engine
----------------------------------------
Hybrid Spam & Phishing Detection System
Trains a Multinomial Naive Bayes classifier on the SMS Spam Collection dataset
using TF-IDF vectorization (max_features=3000) and saves the serialized artifacts.
"""

import os
import re
import string
import joblib
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Public raw URL for the SMS Spam Collection dataset
DATASET_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
MODEL_PATH = "spam_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"


def download_nltk_resources() -> set:
    """Download NLTK stopwords if not already present and return as a set."""
    print("[1/5] Ensuring NLTK stopwords are downloaded...")
    nltk.download("stopwords", quiet=True)
    return set(stopwords.words("english"))


def clean_text(text: str, stop_words: set) -> str:
    """
    Cleans raw input text:
    1. Lowercases the string
    2. Strips punctuation marks
    3. Removes standard English stopwords
    """
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Tokenize and remove stopwords
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)


def load_dataset(url: str) -> pd.DataFrame:
    """Downloads and loads the TSV dataset into a pandas DataFrame."""
    print(f"[2/5] Downloading dataset from public repository: {url}...")
    try:
        df = pd.read_csv(
            url,
            sep="\t",
            header=None,
            names=["label", "message"],
            encoding="utf-8"
        )
        print(f"      Successfully loaded {len(df)} samples ({df['label'].value_counts().to_dict()})")
        return df
    except Exception as exc:
        raise RuntimeError(f"Failed to load dataset from {url}: {exc}") from exc


def main():
    print("=" * 65)
    print(" Hybrid Spam & Phishing Detection - Training ML Engine ")
    print("=" * 65)

    # 1. Download stopwords
    stop_words = download_nltk_resources()

    # 2. Download and load dataset
    df = load_dataset(DATASET_URL)

    # 3. Clean dataset text
    print("[3/5] Cleaning text (lowercasing, removing punctuation & stopwords)...")
    df["clean_message"] = df["message"].apply(lambda msg: clean_text(msg, stop_words))

    # 80/20 train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_message"],
        df["label"],
        test_size=0.20,
        random_state=42,
        stratify=df["label"]
    )
    print(f"      Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    # 4. TF-IDF Vectorization
    print("[4/5] Vectorizing text with TfidfVectorizer (max_features=3000)...")
    vectorizer = TfidfVectorizer(max_features=3000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train Multinomial Naive Bayes classifier
    print("      Training Multinomial Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # 5. Evaluation and Persistence
    print("[5/5] Evaluating model performance...")
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n" + "-" * 40)
    print(f" Test Accuracy: {accuracy * 100:.2f}%")
    print("-" * 40)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save artifacts using joblib
    print(f"\nSaving model to '{MODEL_PATH}'...")
    joblib.dump(model, MODEL_PATH)

    print(f"Saving vectorizer to '{VECTORIZER_PATH}'...")
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("=" * 65)
    print(" Training completed successfully! Ready for inference.")
    print("=" * 65)


if __name__ == "__main__":
    main()
