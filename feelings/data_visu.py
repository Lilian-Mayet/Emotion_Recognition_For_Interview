import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
import seaborn as sns

# Charger le fichier labels.csv
labels_path = 'dataset/labels.csv'  # Remplacez par le chemin réel de labels.csv
data = pd.read_csv(labels_path)

# Afficher les premières lignes du jeu de données
print("Aperçu du jeu de données :")
print(data.head())

# Statistiques de base
print("\nStatistiques descriptives du dataset :")
print(data.describe())

# Vérifier les valeurs manquantes
missing_values = data.isnull().sum()
print("\nValeurs manquantes par colonne :")
print(missing_values)

# Nombre d'images par label
label_counts = data['label'].value_counts()
print("\nNombre d'images par label :")
print(label_counts)

# Vérifier les chemins des images valides
invalid_paths = data[~data['pth'].apply(os.path.exists)]
print(f"\nNombre de chemins d'images invalides : {len(invalid_paths)}")

# Distribution des tailles d'images
image_sizes = []


# Analyser les dimensions d'images
if image_sizes:
    widths, heights = zip(*image_sizes)
    print("\nDimensions des images :")
    print(f"Largeur moyenne : {sum(widths)/len(widths):.2f}")
    print(f"Hauteur moyenne : {sum(heights)/len(heights):.2f}")
else:
    print("\nPas de dimensions d'images disponibles.")

# Graphiques
plt.figure(figsize=(10, 6))
sns.barplot(x=label_counts.index, y=label_counts.values, palette="viridis")
plt.title("Distribution des labels")
plt.ylabel("Nombre d'images")
plt.xlabel("Labels")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(widths, bins=20, kde=True, color='blue')
plt.title("Distribution des largeurs d'images")
plt.xlabel("Largeur")
plt.ylabel("Fréquence")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(heights, bins=20, kde=True, color='orange')
plt.title("Distribution des hauteurs d'images")
plt.xlabel("Hauteur")
plt.ylabel("Fréquence")
plt.tight_layout()
plt.show()

# Afficher la corrélation entre dimensions si possible
if image_sizes:
    image_df = pd.DataFrame({'Width': widths, 'Height': heights})
    print("\nCorrélation entre largeur et hauteur des images :")
    print(image_df.corr())

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=image_df['Width'], y=image_df['Height'], alpha=0.6)
    plt.title("Corrélation entre largeur et hauteur des images")
    plt.xlabel("Largeur")
    plt.ylabel("Hauteur")
    plt.tight_layout()
    plt.show()

# Exporter un rapport d'analyse
report_path = "dataset_analysis_report.csv"
data_summary = {
    "Total Images": [len(data)],
    "Missing Labels": [missing_values['label']],
    "Invalid Paths": [len(invalid_paths)],
    "Unique Labels": [len(label_counts)],
    "Avg Width": [sum(widths) / len(widths) if widths else None],
    "Avg Height": [sum(heights) / len(heights) if heights else None],
}
report_df = pd.DataFrame(data_summary)
report_df.to_csv(report_path, index=False)
print(f"\nRapport d'analyse enregistré sous : {report_path}")
