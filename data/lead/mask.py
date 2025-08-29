import os
import numpy as np
from PIL import Image

mask_dir = "masks"

for f in sorted(os.listdir(mask_dir))[:50]:  # On teste 50 masques
    path = os.path.join(mask_dir, f)
    mask = np.array(Image.open(path))

    uniques = np.unique(mask)
    print(f"{f} => classes: {uniques}")

