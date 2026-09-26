import gzip
import json
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
from pathlib import Path

# NLTK Ressourcen herunterladen (einmalig erforderlich)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


def load_mcauley_json_gz(file_path: str, max_samples: int = None) -> pd.DataFrame:
    """
    Lädt eine gzip-komprimierte JSONL-Datei des McAuley Amazon Reviews Datensatzes.
    
    Erwartet relevante Felder z.B.:
    - 'rating' oder 'overall': Numerische Bewertung (1-5)
    """
    data = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if max_samples and idx >= max_samples:
                break
            entry = json.loads(line)
            
            # Schlüssel an Datensatzstruktur anpassen (z. B. 'rating' vs. 'overall')
            rating = entry.get('rating', entry.get('overall'))
            
            data.append({
                'rating': rating
            })
            
    df = pd.DataFrame(data)
    return df


def main():
    data_dir = Path('data/')
    for file_name in [f for f in os.listdir(data_dir) if f[-9:]=='.jsonl.gz']:
        file_path = data_dir / file_name


        # ---------------------------------------------------------
        # 1. Daten laden (Pfade zu deinen heruntergeladenen Dateien anpassen)
        # ---------------------------------------------------------
        print("=" * 60)
        print(f"Lade Daten aus {file_path}...")
        print("=" * 60)
        
        # max_samples kann zum schnellen Testen genutzt werden (z. B. 10000)
        df = load_mcauley_json_gz(file_path)
        print(f"Geladene Zeilen: {len(df)}")
            
        y = df['rating'].astype(int)
        
        print("\nKlassenverteilung im Datensatz:")
        print(y.value_counts().sort_index())

if __name__ == "__main__":
    main()