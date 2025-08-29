import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Utilise un backend non interactif (pas besoin d'affichage)
import torch
from PIL import Image
from dataset.lead import MultiClassSegmentation2D
from easydict import EasyDict as edict

# Configuration
args = edict({
    'image_size': 256,
    'out_size': 256,
    'data_path': './data/lead'
})

# Créer le dataset
dataset = MultiClassSegmentation2D(args, args.data_path, mode='Training')

print(f"Dataset size: {len(dataset)}")
print(f"Expected classes: [0, 50, 100, 150, 200, 250]")
print("="*60)

# Statistiques globales
all_classes = set()
empty_masks = 0

for i in range(min(10, len(dataset))):  # Analyser les 10 premiers
    try:
        sample = dataset[i]
        img = sample['image']
        mask = sample['label'].squeeze().numpy() if isinstance(sample['label'], torch.Tensor) else sample['label']
        pt = sample['pt']
        p_label = sample['p_label']
        img_name = sample['img_name'] if 'img_name' in sample else f'sample_{i}'  # <- ajoute ça ici

        unique_classes = np.unique(mask)
        all_classes.update(unique_classes)
        
        if len(unique_classes) <= 1:
            empty_masks += 1
        
        # Affichage détaillé
        plt.figure(figsize=(15, 4))
        
        # Image originale avec point de click
        plt.subplot(1, 4, 1)
        if isinstance(img, torch.Tensor):
            img_np = img.permute(1, 2, 0).numpy()
            if img_np.max() <= 1.0:  # Normalisé entre 0-1
                img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = np.array(img)
        
        plt.imshow(img_np)
        plt.title(f"Image {i}")
        plt.scatter(pt[0], pt[1], c='red', s=50, marker='x')
        plt.text(pt[0]+5, pt[1]+5, f"label={p_label}", color='red', fontsize=8)
        plt.axis("off")
        
        # Masque original
        plt.subplot(1, 4, 2)
        plt.imshow(mask, cmap='tab10', vmin=0, vmax=250)
        plt.title(f"Mask (classes: {unique_classes})")
        plt.colorbar(shrink=0.6)
        plt.axis("off")
        
        # Masque binaire de la classe sélectionnée
        plt.subplot(1, 4, 3)
        if len(unique_classes) > 1:
            # Trouver quelle classe a été sélectionnée près du point de click
            y, x = int(pt[1]), int(pt[0])
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                selected_class = mask[y, x]
                binary_mask = (mask == selected_class).astype(np.uint8)
                plt.imshow(binary_mask, cmap='gray')
                plt.title(f"Selected class: {selected_class}")
            else:
                plt.imshow(np.zeros_like(mask), cmap='gray')
                plt.title("Invalid point location")
        else:
            plt.imshow(np.zeros_like(mask), cmap='gray')
            plt.title("No foreground classes")
        plt.axis("off")
        
        # Distribution des valeurs
        plt.subplot(1, 4, 4)
        plt.hist(mask.ravel(), bins=50, alpha=0.7)
        plt.title("Pixel Value Distribution")
        plt.xlabel("Pixel Value")
        plt.ylabel("Count")
        
        plt.tight_layout()
        plt.savefig(f'debug_sample_{i}.png', dpi=100, bbox_inches='tight')
        #plt.show()
        
        # Statistiques détaillées
        print(f"Sample {i}:")
        print(f"  Point: {pt}, Label: {p_label}")
        print(f"  Unique classes: {unique_classes}")
        print(f"  Class distribution: {[(cls, np.sum(mask == cls)) for cls in unique_classes]}")
        
        # Vérifications importantes
        if len(unique_classes) <= 1:
            print(f"  WARNING: Only background class found!")
        
        y, x = int(pt[1]), int(pt[0])
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            clicked_value = mask[y, x]
            print(f"  Clicked pixel value: {clicked_value}")
            if clicked_value == 0:
                print(f"  WARNING: Clicked on background!")
        else:
            print(f"  ERROR: Point outside mask bounds!")
        
        print("-" * 40)
        
    except Exception as e:
        print(f"Error processing sample {i}: {repr(e)}")
        continue

# Résumé global
print("\n" + "="*60)
print(" GLOBAL ANALYSIS:")
print(f" All classes found: {sorted(all_classes)}")
print(f" Empty masks: {empty_masks}/{min(10, len(dataset))}")

# Vérifications critiques
expected_classes = {0, 50, 100, 150, 200, 250}
found_classes = set(all_classes)

if found_classes == expected_classes:
    print("All expected classes found!")
elif found_classes.issubset(expected_classes):
    missing = expected_classes - found_classes
    print(f" Missing classes: {missing}")
else:
    unexpected = found_classes - expected_classes
    print(f" Unexpected classes found: {unexpected}")

# Recommandations
print("\n RECOMMENDATIONS:")
if empty_masks > 0:
    print(f"  - {empty_masks} masks have no foreground classes")
if 0 not in found_classes:
    print("  - No background class (0) found - this might be problematic")
if len(found_classes) < 3:
    print("  - Very few classes detected - check your mask generation")

print("\nRun this script to identify data issues before training!")
