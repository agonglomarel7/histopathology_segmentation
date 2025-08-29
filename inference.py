# -*- coding: utf-8 -*-

"""
usage example:
python MedSAM_Inference.py -i assets/img_demo.png -o ./ --box "[95,255,190,350]"

"""

# %% load environment
import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
import argparse


# visualization functions
# source: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
# change color to avoid red and green
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


# %% load model and image
parser = argparse.ArgumentParser(
    description="run inference on testing set based on MedSAM"
)
parser.add_argument(
    "-i",
    "--data_path",
    type=str,
    default="data/lead/images/16P9867M1.jpg",
    help="path to the data folder",
)
parser.add_argument(
    "-o",
    "--seg_path",
    type=str,
    default="assets/",
    help="path to the segmentation folder",
)
parser.add_argument(
    "--box",
    type=str,
    default='[95, 255, 190, 350]',
    help="bounding box of the segmentation target",
)
parser.add_argument("--device", type=str, default="cuda:0", help="device")
parser.add_argument(
    "-chk",
    "--checkpoint",
    type=str,
    default="logs/msa_lead_test_2025_08_29_12_42_34/Model/checkpoint_best.pth",
    help="path to the trained model",
)
args = parser.parse_args()

# %% load model and checkpoint
device = args.device

print(f"Loading checkpoint from {args.checkpoint}...")
# Charger le checkpoint
ckpt = torch.load(args.checkpoint, map_location="cpu")
print(f"Checkpoint keys: {list(ckpt.keys())}")

# Récupérer uniquement les poids du modèle
if "state_dict" in ckpt:
    state_dict = ckpt["state_dict"]
    print("Using 'state_dict' from checkpoint")
elif "model" in ckpt:
    state_dict = ckpt["model"]
    print("Using 'model' from checkpoint")
else:
    state_dict = ckpt
    print("Using checkpoint directly as state_dict")

# Initialiser un SAM vide (sans checkpoint pré-entraîné)
print("Initializing SAM model...")
medsam_model = sam_model_registry["vit_b"](checkpoint=None)

# Charger les poids
print("Loading state dict...")
missing, unexpected = medsam_model.load_state_dict(state_dict, strict=False)
print(f"Missing keys: {len(missing)}")
print(f"Unexpected keys: {len(unexpected)}")
if missing:
    print("Missing keys:", missing[:5], "..." if len(missing) > 5 else "")
if unexpected:
    print("Unexpected keys:", unexpected[:5], "..." if len(unexpected) > 5 else "")

medsam_model = medsam_model.to(device)
medsam_model.eval()
print("Model loaded successfully!")

# Load and preprocess image
print(f"Loading image from {args.data_path}...")
img_np = io.imread(args.data_path)
if len(img_np.shape) == 2:
    img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
else:
    img_3c = img_np
H, W, _ = img_3c.shape
print(f"Image shape: {img_3c.shape}")

# %% image preprocessing
img_1024 = transform.resize(
    img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
).astype(np.uint8)
img_1024 = (img_1024 - img_1024.min()) / np.clip(
    img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
)  # normalize to [0, 1], (H, W, 3)
# convert the shape to (3, H, W)
img_1024_tensor = (
    torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
)

# Parse bounding box
box_np = np.array([[int(x) for x in args.box[1:-1].split(',')]]) 
print(f"Bounding box (original): {box_np[0]}")
# transfer box_np to 1024x1024 scale
box_1024 = box_np / np.array([W, H, W, H]) * 1024
print(f"Bounding box (1024 scale): {box_1024[0]}")

# Generate image embeddings
print("Generating image embeddings...")
with torch.no_grad():
    image_embedding = medsam_model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)

# Run inference
print("Running MedSAM inference...")
medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, H, W)

# Save segmentation result
output_path = join(args.seg_path, "seg_" + os.path.basename(args.data_path))
io.imsave(output_path, medsam_seg, check_contrast=False)
print(f"Segmentation saved to: {output_path}")

# %% visualize results
print("Creating visualization...")
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img_3c)
show_box(box_np[0], ax[0])
ax[0].set_title("Input Image and Bounding Box")
ax[1].imshow(img_3c)
show_mask(medsam_seg, ax[1])
show_box(box_np[0], ax[1])
ax[1].set_title("MedSAM Segmentation")
plt.tight_layout()
plt.savefig(join(args.seg_path, "visualization_" + os.path.basename(args.data_path).replace('.jpg', '.png')))
plt.show()
print("Done!")
