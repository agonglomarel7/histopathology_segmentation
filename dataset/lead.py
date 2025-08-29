import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils import random_click

class MultiClassSegmentation2D(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='Training', prompt='click'):
        self.data_path = data_path
        self.image_dir = os.path.join(data_path, 'images')
        self.mask_dir = os.path.join(data_path, 'masks')
        
        csv_path = os.path.join(data_path, "dataset_paths.csv")
        df = pd.read_csv(csv_path)
        self.name_list = df.iloc[:, 0].tolist()
        self.label_list = df.iloc[:, 1].tolist()

        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):   # <-- doit être dans la classe
        point_label = 1  # Positive prompt par défaut

        name = self.name_list[index]
        mask_name = self.label_list[index]

        img_path = os.path.join(self.image_dir, name)
        msk_path = os.path.join(self.mask_dir, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')
        mask = mask.resize((self.img_size, self.img_size), resample=Image.NEAREST)
        np_mask = np.array(mask)

        # Générer un point de clic aléatoire sur une classe > 0
        if self.prompt == 'click':
            foreground_classes = np.unique(np_mask)
            foreground_classes = foreground_classes[foreground_classes > 0]

            if len(foreground_classes) > 0:
                chosen_class = np.random.choice(foreground_classes)
                binary_mask = (np_mask == chosen_class).astype(np.uint8)
                point_label, pt = random_click(binary_mask, point_label)
            else:
                pt = (self.img_size // 2, self.img_size // 2)
        else:
            pt = (self.img_size // 2, self.img_size // 2)

        # Appliquer les transformations
        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)

        if self.transform_msk:
            mask = self.transform_msk(mask).long()

        image_meta_dict = {'filename_or_obj': os.path.splitext(os.path.basename(name))[0]}

        return {
            'image': img,
            'label': mask,
            'p_label': point_label,
            'pt': pt,
            'image_meta_dict': image_meta_dict,
            'img_name': name
        }

