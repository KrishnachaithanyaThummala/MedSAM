# segment_anything/build_sam.py

```python
   1 | # -*- coding: utf-8 -*-
   2 | # Copyright (c) Meta Platforms, Inc. and affiliates.
   3 | # All rights reserved.
   4 | 
   5 | # This source code is licensed under the license found in the
   6 | # LICENSE file in the root directory of this source tree.
   7 | from functools import partial
   8 | from pathlib import Path
   9 | import urllib.request
  10 | import torch
  11 | 
  12 | from .modeling import (
  13 |     ImageEncoderViT,
  14 |     MaskDecoder,
  15 |     PromptEncoder,
  16 |     Sam,
  17 |     TwoWayTransformer,
  18 | )
  19 | 
  20 | 
  21 | def build_sam_vit_h(checkpoint=None):
  22 |     return _build_sam(
  23 |         encoder_embed_dim=1280,
  24 |         encoder_depth=32,
  25 |         encoder_num_heads=16,
  26 |         encoder_global_attn_indexes=[7, 15, 23, 31],
  27 |         checkpoint=checkpoint,
  28 |     )
  29 | 
  30 | 
  31 | build_sam = build_sam_vit_h
  32 | 
  33 | 
  34 | def build_sam_vit_l(checkpoint=None):
  35 |     return _build_sam(
  36 |         encoder_embed_dim=1024,
  37 |         encoder_depth=24,
  38 |         encoder_num_heads=16,
  39 |         encoder_global_attn_indexes=[5, 11, 17, 23],
  40 |         checkpoint=checkpoint,
  41 |     )
  42 | 
  43 | 
  44 | def build_sam_vit_b(checkpoint=None):
  45 |     return _build_sam(
  46 |         encoder_embed_dim=768,
  47 |         encoder_depth=12,
  48 |         encoder_num_heads=12,
  49 |         encoder_global_attn_indexes=[2, 5, 8, 11],
  50 |         checkpoint=checkpoint,
  51 |     )
  52 | 
  53 | 
  54 | sam_model_registry = {
  55 |     "default": build_sam_vit_h,
  56 |     "vit_h": build_sam_vit_h,
  57 |     "vit_l": build_sam_vit_l,
  58 |     "vit_b": build_sam_vit_b,
  59 | }
  60 | 
  61 | 
  62 | def _build_sam(
  63 |     encoder_embed_dim,
  64 |     encoder_depth,
  65 |     encoder_num_heads,
  66 |     encoder_global_attn_indexes,
  67 |     checkpoint=None,
  68 | ):
  69 |     prompt_embed_dim = 256
  70 |     image_size = 1024
  71 |     vit_patch_size = 16
  72 |     image_embedding_size = image_size // vit_patch_size
  73 |     sam = Sam(
  74 |         image_encoder=ImageEncoderViT(
  75 |             depth=encoder_depth,
  76 |             embed_dim=encoder_embed_dim,
  77 |             img_size=image_size,
  78 |             mlp_ratio=4,
  79 |             norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
  80 |             num_heads=encoder_num_heads,
  81 |             patch_size=vit_patch_size,
  82 |             qkv_bias=True,
  83 |             use_rel_pos=True,
  84 |             global_attn_indexes=encoder_global_attn_indexes,
  85 |             window_size=14,
  86 |             out_chans=prompt_embed_dim,
  87 |         ),
  88 |         prompt_encoder=PromptEncoder(
  89 |             embed_dim=prompt_embed_dim,
  90 |             image_embedding_size=(image_embedding_size, image_embedding_size),
  91 |             input_image_size=(image_size, image_size),
  92 |             mask_in_chans=16,
  93 |         ),
  94 |         mask_decoder=MaskDecoder(
  95 |             num_multimask_outputs=3,
  96 |             transformer=TwoWayTransformer(
  97 |                 depth=2,
  98 |                 embedding_dim=prompt_embed_dim,
  99 |                 mlp_dim=2048,
 100 |                 num_heads=8,
 101 |             ),
 102 |             transformer_dim=prompt_embed_dim,
 103 |             iou_head_depth=3,
 104 |             iou_head_hidden_dim=256,
 105 |         ),
 106 |         pixel_mean=[123.675, 116.28, 103.53],
 107 |         pixel_std=[58.395, 57.12, 57.375],
 108 |     )
 109 |     sam.eval()
 110 |     checkpoint = Path(checkpoint)
 111 |     if checkpoint.name == "sam_vit_b_01ec64.pth" and not checkpoint.exists():
 112 |         cmd = input("Download sam_vit_b_01ec64.pth from facebook AI? [y]/n: ")
 113 |         if len(cmd) == 0 or cmd.lower() == "y":
 114 |             checkpoint.parent.mkdir(parents=True, exist_ok=True)
 115 |             print("Downloading SAM ViT-B checkpoint...")
 116 |             urllib.request.urlretrieve(
 117 |                 "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
 118 |                 checkpoint,
 119 |             )
 120 |             print(checkpoint.name, " is downloaded!")
 121 |     elif checkpoint.name == "sam_vit_h_4b8939.pth" and not checkpoint.exists():
 122 |         cmd = input("Download sam_vit_h_4b8939.pth from facebook AI? [y]/n: ")
 123 |         if len(cmd) == 0 or cmd.lower() == "y":
 124 |             checkpoint.parent.mkdir(parents=True, exist_ok=True)
 125 |             print("Downloading SAM ViT-H checkpoint...")
 126 |             urllib.request.urlretrieve(
 127 |                 "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
 128 |                 checkpoint,
 129 |             )
 130 |             print(checkpoint.name, " is downloaded!")
 131 |     elif checkpoint.name == "sam_vit_l_0b3195.pth" and not checkpoint.exists():
 132 |         cmd = input("Download sam_vit_l_0b3195.pth from facebook AI? [y]/n: ")
 133 |         if len(cmd) == 0 or cmd.lower() == "y":
 134 |             checkpoint.parent.mkdir(parents=True, exist_ok=True)
 135 |             print("Downloading SAM ViT-L checkpoint...")
 136 |             urllib.request.urlretrieve(
 137 |                 "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
 138 |                 checkpoint,
 139 |             )
 140 |             print(checkpoint.name, " is downloaded!")
 141 | 
 142 |     if checkpoint is not None:
 143 |         with open(checkpoint, "rb") as f:
 144 |             state_dict = torch.load(f)
 145 |         sam.load_state_dict(state_dict)
 146 |     return sam
```
