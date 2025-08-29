import pandas as pd

# Charger le fichier
csv_path = "dataset_paths.csv"
df = pd.read_csv(csv_path)

# Supposer que les colonnes sont : 'image' et 'mask' (ou colonnes 0 et 1)
df.iloc[:, 0] = df.iloc[:, 0].str.replace(r'^images/', '', regex=True)
df.iloc[:, 1] = df.iloc[:, 1].str.replace(r'^masks/', '', regex=True)

# Sauvegarder dans le même fichier (ou un nouveau si tu préfères)
df.to_csv(csv_path, index=False)

print(" Préfixes supprimés et CSV mis à jour.")

