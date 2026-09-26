import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # 1. Daten laden
    file_path = "data/sampled_amazon_reviews.parquet"
    print(f"Lade Datensatz '{file_path}'...")
    df = pd.read_parquet(file_path)

    # Text & Titel kombinieren
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    df['full_text'] = (df['title'] + " " + df['text']).str.strip()

    X = df['full_text']
    y = df['rating'].astype(int)

    # 2. Stratified Train-Test-Split (80% Train, 20% Test)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Trainingsdaten: {len(X_train_raw)} | Testdaten: {len(X_test_raw)}")

    # 3. TF-IDF Vektorisierung
    print("\nErstelle TF-IDF Features...")
    tfidf = TfidfVectorizer(
        max_features=10000, 
        ngram_range=(1, 2), 
        stop_words='english'
    )
    
    # Fit NUR auf den Trainingsdaten (Data Leakage vermeiden)
    X_train = tfidf.fit_transform(X_train_raw)
    X_test = tfidf.transform(X_test_raw)

    # 4. Modell 1: Logistische Regression
    print("\n============================================================")
    print("Trainiere Modell 1: Logistic Regression (class_weight='balanced')")
    print("============================================================")
    log_reg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    log_reg.fit(X_train, y_train)

    y_pred_lr = log_reg.predict(X_test)
    
    print(f"Accuracy (LogReg): {accuracy_score(y_test, y_pred_lr):.4f}")
    print("\nDetaillierter Report:")
    print(classification_report(y_test, y_pred_lr))

    # 5. Modell 2: Multinomial Naive Bayes (zum Vergleich)
    print("\n============================================================")
    print("Trainiere Modell 2: Multinomial Naive Bayes")
    print("============================================================")
    nb = MultinomialNB()
    nb.fit(X_train, y_train)

    y_pred_nb = nb.predict(X_test)
    
    print(f"Accuracy (Naive Bayes): {accuracy_score(y_test, y_pred_nb):.4f}")
    print("\nDetaillierter Report:")
    print(classification_report(y_test, y_pred_nb))

    # 6. Confusion Matrix für Logistische Regression anzeigen/speichern
    cm = confusion_matrix(y_test, y_pred_lr)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[1, 2, 3, 4, 5], 
                yticklabels=[1, 2, 3, 4, 5])
    plt.title('Confusion Matrix - Logistic Regression')
    plt.xlabel('Vorhergesagtes Rating')
    plt.ylabel('Tatsächliches Rating')
    plt.tight_layout()
    plt.savefig('confusion_matrix_logreg.png')
    print("\nConfusion Matrix als 'confusion_matrix_logreg.png' gespeichert.")

if __name__ == "__main__":
    main()