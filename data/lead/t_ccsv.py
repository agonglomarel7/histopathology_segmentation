# -*- coding: utf-8 -*-
import os
import csv

# Dossiers à adapter
images_dir = "images/"
masks_dir = "masks/"
output_csv = "dataset_paths.csv"

# Récupère toutes les images .jpg
image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]

# Préparation des lignes valides
rows = []

for img in image_files:
    base_name = os.path.splitext(img)[0]
    mask_name = f"{base_name}_mask.png"
    mask_path = os.path.join(masks_dir, mask_name)
    image_path = os.path.join(images_dir, img)

    if os.path.isfile(mask_path):
        rows.append([image_path, mask_path])
    else:
        print(f" Mask manquant pour : {img}")

# Écriture du CSV
with open(output_csv, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "mask_path"])
    writer.writerows(rows)

print(f"\n CSV généré : {output_csv} ({len(rows)} paires valides)")

