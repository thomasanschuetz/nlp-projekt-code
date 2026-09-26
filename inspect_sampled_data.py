import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# --- STYLING FÜR WISSENSCHAFTLICHE ARBEITEN ---
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 14
})

def check_sampling(file_path: str = "data/sampled_amazon_reviews.parquet", output_dir: str = "output/figures"):
    print(f"Lade '{file_path}'...\n")
    df = pd.read_parquet(file_path)

    print("============================================================")
    print(f"Gesamtzahl der Zeilen im Datensatz: {len(df)}")
    print("============================================================\n")

    print("1. Anzahl der Rezensionen pro Kategorie:")
    print(df['category'].value_counts())
    print("\n------------------------------------------------------------\n")

    print("2. Verteilung der Ratings (1 bis 5 Sterne) pro Kategorie:")
    rating_matrix = pd.crosstab(df['category'], df['rating'], margins=True, margins_name="Gesamt")
    print(rating_matrix)
    print("\n------------------------------------------------------------\n")

    print("3. Prozentuale Rating-Verteilung innerhalb jeder Kategorie:")
    rating_pct = pd.crosstab(df['category'], df['rating'], normalize='index') * 100
    print(rating_pct.round(2).astype(str) + " %")
    print("\n------------------------------------------------------------\n")

    # --- 4. GRAFIK ERSTELLEN UND SPEICHERN ---
    print("Erstelle und speichere Grafik zur Rating-Verteilung (Train vs. Test)...")

    # Stratified 80/20 Train/Test-Split durchführen
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['rating']
    )

    # Absolute Häufigkeiten berechnen
    train_counts = df_train['rating'].value_counts().sort_index()
    test_counts = df_test['rating'].value_counts().sort_index()

    df_plot = pd.DataFrame({
        "Sterne": train_counts.index,
        "Trainingssatz (80%)": train_counts.values,
        "Testsatz (20%)": test_counts.values
    }).melt(id_vars="Sterne", var_name="Datensatz", value_name="Anzahl Rezensionen")

    # Diagramm zeichnen
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = ["#1f77b4", "#ff7f0e"]  # Blau (Train), Orange (Test)

    barplot = sns.barplot(
        data=df_plot,
        x="Sterne",
        y="Anzahl Rezensionen",
        hue="Datensatz",
        palette=palette,
        ax=ax,
        edgecolor="black",
        linewidth=0.8
    )

    # Werte über den Balken beschriften
    for p in barplot.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f'{int(height):,}',
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom',
                fontsize=9,
                xytext=(0, 3),
                textcoords='offset points'
            )

    # Achsen und Titel definieren
    ax.set_title("Verteilung der Sterne-Bewertungen im Korpus (Class Imbalance)", pad=15, fontweight="bold")
    ax.set_xlabel("Bewertung (Sterne)")
    ax.set_ylabel("Anzahl der Rezensionen")
    ax.set_xticklabels(["1 Stern", "2 Sterne", "3 Sterne", "4 Sterne", "5 Sterne"])
    ax.set_ylim(0, max(df_plot["Anzahl Rezensionen"]) * 1.15)
    ax.legend(title="Datensatz", frameon=True, facecolor="white")

    plt.tight_layout()

    # Ordner erstellen und Grafik speichern
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, "rating_distribution.png")
    pdf_path = os.path.join(output_dir, "rating_distribution.pdf")

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"Grafik erfolgreich gespeichert unter:\n - {png_path}\n - {pdf_path}\n")

if __name__ == "__main__":
    check_sampling()