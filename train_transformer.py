import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)

def compute_metrics(eval_pred):
    """Berechnet Accuracy und Macro F1-Score für das Evaluation-Set."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    
    return {
        'accuracy': acc,
        'f1_macro': f1,
        'precision_macro': precision,
        'recall_macro': recall
    }

def main():
    MODEL_NAME = "distilbert-base-uncased"
    FILE_PATH = "sampled_amazon_reviews.parquet"
    
    # 1. Daten laden & vorbereiten
    print("Lade Daten...")
    df = pd.read_parquet(FILE_PATH)
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    df['full_text'] = (df['title'] + " " + df['text']).str.strip()
    
    # HuggingFace erwartet Labels von 0 bis 4 (statt 1 bis 5)
    df['label'] = df['rating'].astype(int) - 1
    
    # Train-Test Split (80% Train, 20% Test)
    train_df, test_df = train_test_split(
        df[['full_text', 'label']], 
        test_size=0.2, 
        random_state=42, 
        stratify=df['label']
    )
    
    print(f"Trainingsgröße: {len(train_df)}, Testgröße: {len(test_df)}")

    # Pandas in Hugging Face Datasets umwandeln
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # 2. Tokenizer laden
    print(f"Lade Tokenizer für '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples['full_text'], truncation=True, max_length=256, padding="max_length")

    print("Tokenisiere Daten...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # 3. Modell laden
    print(f"Lade Modell '{MODEL_NAME}'...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=5
    )

    # 4. Training Arguments definieren
    training_args = TrainingArguments(
        output_dir="./results_distilbert",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        fp16=torch.cuda.is_available(),  # Nutzt Gemischte Präzision, falls GPU vorhanden
        logging_steps=100,
    )

    # 5. Trainer initialisieren
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 6. Training starten
    print("Starte Modell-Training...")
    trainer.train()

    # 7. Finales Evaluation-Ergebnis ausgeben
    print("\n============================================================")
    print("Finales Evaluation-Ergebnis auf dem Testset:")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    main()