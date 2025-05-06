import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Chemins des dossiers et fichiers
original_data_dir = "dataset"
train_dir = "data/train"
val_dir = "data/val"
labels_csv_path = "dataset/labels.csv"

def split_dataset_with_labels(original_data_dir, train_dir, val_dir, labels_csv_path, test_size=0.2):
    # Charger le fichier labels.csv
    if not os.path.exists(labels_csv_path):
        raise FileNotFoundError(f"Le fichier {labels_csv_path} est introuvable.")
    
    labels_df = pd.read_csv(labels_csv_path)

    # Créer les dossiers de sortie
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Créer des DataFrames pour train et val
    train_data = pd.DataFrame(columns=labels_df.columns)
    val_data = pd.DataFrame(columns=labels_df.columns)

    # Parcourir les labels et diviser en train/val
    for label in labels_df["label"].unique():
        # Filtrer les images pour ce label
        label_data = labels_df[labels_df["label"] == label]

        # Diviser en train/val
        train_split, val_split = train_test_split(label_data, test_size=test_size, random_state=42)

        # Ajouter aux DataFrames respectifs
        train_data = pd.concat([train_data, train_split], ignore_index=True)
        val_data = pd.concat([val_data, val_split], ignore_index=True)

        # Copier les fichiers dans les répertoires correspondants
        for _, row in train_split.iterrows():
            src_path = original_data_dir+"/"+ row["pth"]
            dest_dir = os.path.join(train_dir, label)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(src_path, os.path.join(dest_dir, os.path.basename(src_path)))

        for _, row in val_split.iterrows():
            src_path = original_data_dir+"/"+ row["pth"]
            dest_dir = os.path.join(val_dir, label)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(src_path, os.path.join(dest_dir, os.path.basename(src_path)))

    # Mettre à jour le fichier labels.csv pour chaque ensemble
    train_csv_path = os.path.join(train_dir, "labels_train.csv")
    val_csv_path = os.path.join(val_dir, "labels_val.csv")
    train_data.to_csv(train_csv_path, index=False)
    val_data.to_csv(val_csv_path, index=False)

    print(f"Dataset divisé avec succès :\n - Train : {train_dir}\n - Validation : {val_dir}")
    print(f"Labels mis à jour :\n - Train CSV : {train_csv_path}\n - Validation CSV : {val_csv_path}")

# Appeler la fonction pour diviser le dataset
split_dataset_with_labels(original_data_dir, train_dir, val_dir, labels_csv_path, test_size=0.2)
