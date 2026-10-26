# segment_anything/modeling/sam.py

```python
   1 | # -*- coding: utf-8 -*-
   2 | # Copyright (c) Meta Platforms, Inc. and affiliates.
   3 | # All rights reserved.
   4 | 
   5 | # This source code is licensed under the license found in the
   6 | # LICENSE file in the root directory of this source tree.
   7 | 
   8 | import torch
   9 | from torch import nn
  10 | from torch.nn import functional as F
  11 | 
  12 | from typing import Any, Dict, List, Tuple
  13 | 
  14 | from .image_encoder import ImageEncoderViT
  15 | from .mask_decoder import MaskDecoder
  16 | from .prompt_encoder import PromptEncoder
  17 | 
  18 | 
  19 | class Sam(nn.Module):
  20 |     mask_threshold: float = 0.0
  21 |     image_format: str = "RGB"
  22 | 
  23 |     def __init__(
  24 |         self,
  25 |         image_encoder: ImageEncoderViT,
  26 |         prompt_encoder: PromptEncoder,
  27 |         mask_decoder: MaskDecoder,
  28 |         pixel_mean: List[float] = [123.675, 116.28, 103.53],
  29 |         pixel_std: List[float] = [58.395, 57.12, 57.375],
  30 |     ) -> None:
  31 |         """
  32 |         SAM predicts object masks from an image and input prompts.
  33 | 
  34 |         Arguments:
  35 |           image_encoder (ImageEncoderViT): The backbone used to encode the
  36 |             image into image embeddings that allow for efficient mask prediction.
  37 |           prompt_encoder (PromptEncoder): Encodes various types of input prompts.
  38 |           mask_decoder (MaskDecoder): Predicts masks from the image embeddings
  39 |             and encoded prompts.
  40 |           pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
  41 |           pixel_std (list(float)): Std values for normalizing pixels in the input image.
  42 |         """
  43 |         super().__init__()
  44 |         self.image_encoder = image_encoder
  45 |         self.prompt_encoder = prompt_encoder
  46 |         self.mask_decoder = mask_decoder
  47 |         self.register_buffer(
  48 |             "pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
  49 |         )
  50 |         self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
  51 | 
  52 |     @property
  53 |     def device(self) -> Any:
  54 |         return self.pixel_mean.device
  55 | 
  56 |     @torch.no_grad()
  57 |     def forward(
  58 |         self,
  59 |         batched_input: List[Dict[str, Any]],
  60 |         multimask_output: bool,
  61 |     ) -> List[Dict[str, torch.Tensor]]:
  62 |         """
  63 |         Predicts masks end-to-end from provided images and prompts.
  64 |         If prompts are not known in advance, using SamPredictor is
  65 |         recommended over calling the model directly.
  66 | 
  67 |         Arguments:
  68 |           batched_input (list(dict)): A list over input images, each a
  69 |             dictionary with the following keys. A prompt key can be
  70 |             excluded if it is not present.
  71 |               'image': The image as a torch tensor in 3xHxW format,
  72 |                 already transformed for input to the model.
  73 |               'original_size': (tuple(int, int)) The original size of
  74 |                 the image before transformation, as (H, W).
  75 |               'point_coords': (torch.Tensor) Batched point prompts for
  76 |                 this image, with shape BxNx2. Already transformed to the
  77 |                 input frame of the model.
  78 |               'point_labels': (torch.Tensor) Batched labels for point prompts,
  79 |                 with shape BxN.
  80 |               'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
  81 |                 Already transformed to the input frame of the model.
  82 |               'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
  83 |                 in the form Bx1xHxW.
  84 |           multimask_output (bool): Whether the model should predict multiple
  85 |             disambiguating masks, or return a single mask.
  86 | 
  87 |         Returns:
  88 |           (list(dict)): A list over input images, where each element is
  89 |             as dictionary with the following keys.
  90 |               'masks': (torch.Tensor) Batched binary mask predictions,
  91 |                 with shape BxCxHxW, where B is the number of input prompts,
  92 |                 C is determined by multimask_output, and (H, W) is the
  93 |                 original size of the image.
  94 |               'iou_predictions': (torch.Tensor) The model's predictions
  95 |                 of mask quality, in shape BxC.
  96 |               'low_res_logits': (torch.Tensor) Low resolution logits with
  97 |                 shape BxCxHxW, where H=W=256. Can be passed as mask input
  98 |                 to subsequent iterations of prediction.
  99 |         """
 100 |         input_images = torch.stack(
 101 |             [self.preprocess(x["image"]) for x in batched_input], dim=0
 102 |         )
 103 |         image_embeddings = self.image_encoder(input_images)
 104 | 
 105 |         outputs = []
 106 |         for image_record, curr_embedding in zip(batched_input, image_embeddings):
 107 |             if "point_coords" in image_record:
 108 |                 points = (image_record["point_coords"], image_record["point_labels"])
 109 |             else:
 110 |                 points = None
 111 |             sparse_embeddings, dense_embeddings = self.prompt_encoder(
 112 |                 points=points,
 113 |                 boxes=image_record.get("boxes", None),
 114 |                 masks=image_record.get("mask_inputs", None),
 115 |             )
 116 |             low_res_masks, iou_predictions = self.mask_decoder(
 117 |                 image_embeddings=curr_embedding.unsqueeze(0),
 118 |                 image_pe=self.prompt_encoder.get_dense_pe(),
 119 |                 sparse_prompt_embeddings=sparse_embeddings,
 120 |                 dense_prompt_embeddings=dense_embeddings,
 121 |                 multimask_output=multimask_output,
 122 |             )
 123 |             masks = self.postprocess_masks(
 124 |                 low_res_masks,
 125 |                 input_size=image_record["image"].shape[-2:],
 126 |                 original_size=image_record["original_size"],
 127 |             )
 128 |             masks = masks > self.mask_threshold
 129 |             outputs.append(
 130 |                 {
 131 |                     "masks": masks,
 132 |                     "iou_predictions": iou_predictions,
 133 |                     "low_res_logits": low_res_masks,
 134 |                 }
 135 |             )
 136 |         return outputs
 137 | 
 138 |     def postprocess_masks(
 139 |         self,
 140 |         masks: torch.Tensor,
 141 |         input_size: Tuple[int, ...],
 142 |         original_size: Tuple[int, ...],
 143 |     ) -> torch.Tensor:
 144 |         """
 145 |         Remove padding and upscale masks to the original image size.
 146 | 
 147 |         Arguments:
 148 |           masks (torch.Tensor): Batched masks from the mask_decoder,
 149 |             in BxCxHxW format.
 150 |           input_size (tuple(int, int)): The size of the image input to the
 151 |             model, in (H, W) format. Used to remove padding.
 152 |           original_size (tuple(int, int)): The original size of the image
 153 |             before resizing for input to the model, in (H, W) format.
 154 | 
 155 |         Returns:
 156 |           (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
 157 |             is given by original_size.
 158 |         """
 159 |         masks = F.interpolate(
 160 |             masks,
 161 |             (self.image_encoder.img_size, self.image_encoder.img_size),
 162 |             mode="bilinear",
 163 |             align_corners=False,
 164 |         )
 165 |         masks = masks[..., : input_size[0], : input_size[1]]
 166 |         masks = F.interpolate(
 167 |             masks, original_size, mode="bilinear", align_corners=False
 168 |         )
 169 |         return masks
 170 | 
 171 |     def preprocess(self, x: torch.Tensor) -> torch.Tensor:
 172 |         """Normalize pixel values and pad to a square input."""
 173 |         # Normalize colors
 174 |         x = (x - self.pixel_mean) / self.pixel_std
 175 | 
 176 |         # Pad
 177 |         h, w = x.shape[-2:]
 178 |         padh = self.image_encoder.img_size - h
 179 |         padw = self.image_encoder.img_size - w
 180 |         x = F.pad(x, (0, padw, 0, padh))
 181 |         return x
```
