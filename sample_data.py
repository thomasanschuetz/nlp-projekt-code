import gzip
import json
import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_sample_category(file_path: str, samples_per_category: int = 10000) -> pd.DataFrame:
    """
    Liest eine .jsonl.gz-Datei ein, extrahiert die Kategorie aus dem Dateinamen
    und zieht ein stratifiziertes Stichproben-Subset.
    """
    # Kategorie-Namen aus dem Dateinamen extrahieren (z. B. 'data/All_Beauty.jsonl.gz' -> 'All_Beauty')
    base_name = os.path.basename(file_path)
    category_name = base_name.split('.')[0]
    
    print(f"Lade Kategorie '{category_name}' aus {file_path}...")
    
    data = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            
            rating = entry.get('rating', entry.get('overall'))
            title = entry.get('title', '')
            text = entry.get('text', entry.get('reviewText', ''))
            
            data.append({
                'title': title,
                'text': text,
                'rating': rating,
                'category': category_name
            })
            
    df = pd.DataFrame(data)
    df = df.dropna(subset=['rating'])
    df['rating'] = df['rating'].astype(int)
    
    print(f"  -> Ursprüngliche Zeilen: {len(df)}")
    
    # Stratifiziertes Samplen (falls genug Daten vorhanden sind)
    if len(df) > samples_per_category:
        df_sampled, _ = train_test_split(
            df, 
            train_size=samples_per_category, 
            stratify=df['rating'], 
            random_state=42
        )
    else:
        df_sampled = df
        
    print(f"  -> Sampled Zeilen: {len(df_sampled)}")
    return df_sampled

def main():
    data_dir = "data"
    output_file = "sampled_amazon_reviews.parquet"  # Parquet ist schneller & kompakter als CSV
    samples_per_cat = 50000
    
    # Alle .jsonl.gz Dateien im Ordner finden
    search_path = os.path.join(data_dir, "*.jsonl.gz")
    gz_files = glob.glob(search_path)
    
    if not gz_files:
        print(f"Keine .jsonl.gz Dateien in '{data_dir}' gefunden!")
        return
    
    df_list = []
    for file_path in gz_files:
        df_cat = load_and_sample_category(file_path, samples_per_category=samples_per_cat)
        df_list.append(df_cat)
        
    # Alle Kategorien in ein einzelnes DataFrame zusammenführen
    final_df = pd.concat(df_list, ignore_index=True)
    
    print("\n============================================================")
    print(f"Gesamt-DataFrame Form: {final_df.shape}")
    print("\nVerteilung der Kategorien im finalen Set:")
    print(final_df['category'].value_counts())
    
    print("\nVerteilung der Sterne im finalen Set:")
    print(final_df['rating'].value_counts().sort_index())
    
    # Speichern des Ergebnisses
    # Hinweis: .parquet ist extrem schnell und behält Datentypen bei. 
    # Alternativ: final_df.to_csv("sampled_amazon_reviews.csv", index=False)
    print(f"\nSpeichere finales DataFrame in '{output_file}'...")
    final_df.to_parquet(output_file, index=False)
    print("Fertig!")

if __name__ == "__main__":
    main()