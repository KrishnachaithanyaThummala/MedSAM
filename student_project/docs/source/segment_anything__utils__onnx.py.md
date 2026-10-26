# segment_anything/utils/onnx.py

```python
   1 | # -*- coding: utf-8 -*-
   2 | # Copyright (c) Meta Platforms, Inc. and affiliates.
   3 | # All rights reserved.
   4 | 
   5 | # This source code is licensed under the license found in the
   6 | # LICENSE file in the root directory of this source tree.
   7 | 
   8 | import torch
   9 | import torch.nn as nn
  10 | from torch.nn import functional as F
  11 | 
  12 | from typing import Tuple
  13 | 
  14 | from ..modeling import Sam
  15 | from .amg import calculate_stability_score
  16 | 
  17 | 
  18 | class SamOnnxModel(nn.Module):
  19 |     """
  20 |     This model should not be called directly, but is used in ONNX export.
  21 |     It combines the prompt encoder, mask decoder, and mask postprocessing of Sam,
  22 |     with some functions modified to enable model tracing. Also supports extra
  23 |     options controlling what information. See the ONNX export script for details.
  24 |     """
  25 | 
  26 |     def __init__(
  27 |         self,
  28 |         model: Sam,
  29 |         return_single_mask: bool,
  30 |         use_stability_score: bool = False,
  31 |         return_extra_metrics: bool = False,
  32 |     ) -> None:
  33 |         super().__init__()
  34 |         self.mask_decoder = model.mask_decoder
  35 |         self.model = model
  36 |         self.img_size = model.image_encoder.img_size
  37 |         self.return_single_mask = return_single_mask
  38 |         self.use_stability_score = use_stability_score
  39 |         self.stability_score_offset = 1.0
  40 |         self.return_extra_metrics = return_extra_metrics
  41 | 
  42 |     @staticmethod
  43 |     def resize_longest_image_size(
  44 |         input_image_size: torch.Tensor, longest_side: int
  45 |     ) -> torch.Tensor:
  46 |         input_image_size = input_image_size.to(torch.float32)
  47 |         scale = longest_side / torch.max(input_image_size)
  48 |         transformed_size = scale * input_image_size
  49 |         transformed_size = torch.floor(transformed_size + 0.5).to(torch.int64)
  50 |         return transformed_size
  51 | 
  52 |     def _embed_points(
  53 |         self, point_coords: torch.Tensor, point_labels: torch.Tensor
  54 |     ) -> torch.Tensor:
  55 |         point_coords = point_coords + 0.5
  56 |         point_coords = point_coords / self.img_size
  57 |         point_embedding = self.model.prompt_encoder.pe_layer._pe_encoding(point_coords)
  58 |         point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)
  59 | 
  60 |         point_embedding = point_embedding * (point_labels != -1)
  61 |         point_embedding = (
  62 |             point_embedding
  63 |             + self.model.prompt_encoder.not_a_point_embed.weight * (point_labels == -1)
  64 |         )
  65 | 
  66 |         for i in range(self.model.prompt_encoder.num_point_embeddings):
  67 |             point_embedding = (
  68 |                 point_embedding
  69 |                 + self.model.prompt_encoder.point_embeddings[i].weight
  70 |                 * (point_labels == i)
  71 |             )
  72 | 
  73 |         return point_embedding
  74 | 
  75 |     def _embed_masks(
  76 |         self, input_mask: torch.Tensor, has_mask_input: torch.Tensor
  77 |     ) -> torch.Tensor:
  78 |         mask_embedding = has_mask_input * self.model.prompt_encoder.mask_downscaling(
  79 |             input_mask
  80 |         )
  81 |         mask_embedding = mask_embedding + (
  82 |             1 - has_mask_input
  83 |         ) * self.model.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
  84 |         return mask_embedding
  85 | 
  86 |     def mask_postprocessing(
  87 |         self, masks: torch.Tensor, orig_im_size: torch.Tensor
  88 |     ) -> torch.Tensor:
  89 |         masks = F.interpolate(
  90 |             masks,
  91 |             size=(self.img_size, self.img_size),
  92 |             mode="bilinear",
  93 |             align_corners=False,
  94 |         )
  95 | 
  96 |         prepadded_size = self.resize_longest_image_size(orig_im_size, self.img_size).to(
  97 |             torch.int64
  98 |         )
  99 |         masks = masks[..., : prepadded_size[0], : prepadded_size[1]]  # type: ignore
 100 | 
 101 |         orig_im_size = orig_im_size.to(torch.int64)
 102 |         h, w = orig_im_size[0], orig_im_size[1]
 103 |         masks = F.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)
 104 |         return masks
 105 | 
 106 |     def select_masks(
 107 |         self, masks: torch.Tensor, iou_preds: torch.Tensor, num_points: int
 108 |     ) -> Tuple[torch.Tensor, torch.Tensor]:
 109 |         # Determine if we should return the multiclick mask or not from the number of points.
 110 |         # The reweighting is used to avoid control flow.
 111 |         score_reweight = torch.tensor(
 112 |             [[1000] + [0] * (self.model.mask_decoder.num_mask_tokens - 1)]
 113 |         ).to(iou_preds.device)
 114 |         score = iou_preds + (num_points - 2.5) * score_reweight
 115 |         best_idx = torch.argmax(score, dim=1)
 116 |         masks = masks[torch.arange(masks.shape[0]), best_idx, :, :].unsqueeze(1)
 117 |         iou_preds = iou_preds[torch.arange(masks.shape[0]), best_idx].unsqueeze(1)
 118 | 
 119 |         return masks, iou_preds
 120 | 
 121 |     @torch.no_grad()
 122 |     def forward(
 123 |         self,
 124 |         image_embeddings: torch.Tensor,
 125 |         point_coords: torch.Tensor,
 126 |         point_labels: torch.Tensor,
 127 |         mask_input: torch.Tensor,
 128 |         has_mask_input: torch.Tensor,
 129 |         orig_im_size: torch.Tensor,
 130 |     ):
 131 |         sparse_embedding = self._embed_points(point_coords, point_labels)
 132 |         dense_embedding = self._embed_masks(mask_input, has_mask_input)
 133 | 
 134 |         masks, scores = self.model.mask_decoder.predict_masks(
 135 |             image_embeddings=image_embeddings,
 136 |             image_pe=self.model.prompt_encoder.get_dense_pe(),
 137 |             sparse_prompt_embeddings=sparse_embedding,
 138 |             dense_prompt_embeddings=dense_embedding,
 139 |         )
 140 | 
 141 |         if self.use_stability_score:
 142 |             scores = calculate_stability_score(
 143 |                 masks, self.model.mask_threshold, self.stability_score_offset
 144 |             )
 145 | 
 146 |         if self.return_single_mask:
 147 |             masks, scores = self.select_masks(masks, scores, point_coords.shape[1])
 148 | 
 149 |         upscaled_masks = self.mask_postprocessing(masks, orig_im_size)
 150 | 
 151 |         if self.return_extra_metrics:
 152 |             stability_scores = calculate_stability_score(
 153 |                 upscaled_masks, self.model.mask_threshold, self.stability_score_offset
 154 |             )
 155 |             areas = (upscaled_masks > self.model.mask_threshold).sum(-1).sum(-1)
 156 |             return upscaled_masks, scores, stability_scores, areas, masks
 157 | 
 158 |         return upscaled_masks, scores, masks
```
