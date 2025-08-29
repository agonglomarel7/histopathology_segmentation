# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from skimage import io, transform
import torch.nn.functional as F
import argparse

# Import direct des modules SAM
from segment_anything.modeling import Sam
from segment_anything.modeling.image_encoder import ImageEncoderViT
from segment_anything.modeling.mask_decoder import MaskDecoder
from segment_anything.modeling.prompt_encoder import PromptEncoder
from segment_anything.modeling.transformer import TwoWayTransformer
from functools import partial
import torch.nn as nn

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

def create_sam_model():
    """Créer l'architecture SAM ViT-B vide"""
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=12,
            embed_dim=768,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11],
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    return sam

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None, boxes=box_torch, masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(
        low_res_pred, size=(H, W), mode="bilinear", align_corners=False,
    )
    low_res_pred = low_res_pred.squeeze().cpu().numpy()
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, default="data/lead/masks/16P9867M2_mask.png")
    parser.add_argument("-o", "--output", type=str, default="./")
    parser.add_argument("--box", type=str, default='[95, 255, 190, 350]')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("-c", "--checkpoint", type=str, 
                       default="logs/msa_lead_test_2025_08_29_12_42_34/Model/checkpoint_best.pth")
    args = parser.parse_args()

    device = args.device
    
    print("Creating SAM architecture...")
    model = create_sam_model()
    
    print(f"Loading your trained weights: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        print("Using 'state_dict' from checkpoint")
    else:
        state_dict = ckpt
        print("Using checkpoint directly")
    
    # Charger tes poids entraînés
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded weights - Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    
    model = model.to(device)
    model.eval()
    
    print(f"Loading image: {args.image}")
    img_np = io.imread(args.image)
    if len(img_np.shape) == 2:
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    else:
        img_3c = img_np
    H, W, _ = img_3c.shape
    
    # Preprocessing
    img_1024 = transform.resize(img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None)
    img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Parse box
    box_np = np.array([[int(x) for x in args.box[1:-1].split(',')]])
    box_1024 = box_np / np.array([W, H, W, H]) * 1024
    print(f"Box: {box_np[0]} -> {box_1024[0]}")
    
    # Inference
    print("Running inference...")
    with torch.no_grad():
        image_embedding = model.image_encoder(img_1024_tensor)
    
    mask = medsam_inference(model, image_embedding, box_1024, H, W)
    
    # Save results
    output_path = os.path.join(args.output, f"seg_{os.path.basename(args.image)}")
    io.imsave(output_path, mask, check_contrast=False)
    print(f"Saved mask: {output_path}")
    
    # Visualize
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img_3c)
    show_box(box_np[0], ax[0])
    ax[0].set_title("Input + Bounding Box")
    
    ax[1].imshow(img_3c)
    show_mask(mask, ax[1])
    show_box(box_np[0], ax[1])
    ax[1].set_title("MedSAM Segmentation")
    
    plt.tight_layout()
    viz_path = os.path.join(args.output, f"viz_{os.path.basename(args.image).replace('.jpg', '.png')}")
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved visualization: {viz_path}")

if __name__ == "__main__":
    main()
