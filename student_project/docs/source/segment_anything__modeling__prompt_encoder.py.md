# segment_anything/modeling/prompt_encoder.py

```python
   1 | # -*- coding: utf-8 -*-
   2 | # Copyright (c) Meta Platforms, Inc. and affiliates.
   3 | # All rights reserved.
   4 | 
   5 | # This source code is licensed under the license found in the
   6 | # LICENSE file in the root directory of this source tree.
   7 | 
   8 | import numpy as np
   9 | import torch
  10 | from torch import nn
  11 | 
  12 | from typing import Any, Optional, Tuple, Type
  13 | 
  14 | from .common import LayerNorm2d
  15 | 
  16 | 
  17 | class PromptEncoder(nn.Module):
  18 |     def __init__(
  19 |         self,
  20 |         embed_dim: int,
  21 |         image_embedding_size: Tuple[int, int],
  22 |         input_image_size: Tuple[int, int],
  23 |         mask_in_chans: int,
  24 |         activation: Type[nn.Module] = nn.GELU,
  25 |     ) -> None:
  26 |         """
  27 |         Encodes prompts for input to SAM's mask decoder.
  28 | 
  29 |         Arguments:
  30 |           embed_dim (int): The prompts' embedding dimension
  31 |           image_embedding_size (tuple(int, int)): The spatial size of the
  32 |             image embedding, as (H, W).
  33 |           input_image_size (int): The padded size of the image as input
  34 |             to the image encoder, as (H, W).
  35 |           mask_in_chans (int): The number of hidden channels used for
  36 |             encoding input masks.
  37 |           activation (nn.Module): The activation to use when encoding
  38 |             input masks.
  39 |         """
  40 |         super().__init__()
  41 |         self.embed_dim = embed_dim
  42 |         self.input_image_size = input_image_size
  43 |         self.image_embedding_size = image_embedding_size
  44 |         self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
  45 | 
  46 |         self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
  47 |         point_embeddings = [
  48 |             nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)
  49 |         ]
  50 |         self.point_embeddings = nn.ModuleList(point_embeddings)
  51 |         self.not_a_point_embed = nn.Embedding(1, embed_dim)
  52 | 
  53 |         self.mask_input_size = (
  54 |             4 * image_embedding_size[0],
  55 |             4 * image_embedding_size[1],
  56 |         )
  57 |         self.mask_downscaling = nn.Sequential(
  58 |             nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
  59 |             LayerNorm2d(mask_in_chans // 4),
  60 |             activation(),
  61 |             nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
  62 |             LayerNorm2d(mask_in_chans),
  63 |             activation(),
  64 |             nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
  65 |         )
  66 |         self.no_mask_embed = nn.Embedding(1, embed_dim)
  67 | 
  68 |     def get_dense_pe(self) -> torch.Tensor:
  69 |         """
  70 |         Returns the positional encoding used to encode point prompts,
  71 |         applied to a dense set of points the shape of the image encoding.
  72 | 
  73 |         Returns:
  74 |           torch.Tensor: Positional encoding with shape
  75 |             1x(embed_dim)x(embedding_h)x(embedding_w)
  76 |         """
  77 |         return self.pe_layer(self.image_embedding_size).unsqueeze(0)
  78 | 
  79 |     def _embed_points(
  80 |         self,
  81 |         points: torch.Tensor,
  82 |         labels: torch.Tensor,
  83 |         pad: bool,
  84 |     ) -> torch.Tensor:
  85 |         """Embeds point prompts."""
  86 |         points = points + 0.5  # Shift to center of pixel
  87 |         if pad:
  88 |             padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
  89 |             padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
  90 |             points = torch.cat([points, padding_point], dim=1)
  91 |             labels = torch.cat([labels, padding_label], dim=1)
  92 |         point_embedding = self.pe_layer.forward_with_coords(
  93 |             points, self.input_image_size
  94 |         )
  95 |         point_embedding[labels == -1] = 0.0
  96 |         point_embedding[labels == -1] += self.not_a_point_embed.weight
  97 |         point_embedding[labels == 0] += self.point_embeddings[0].weight
  98 |         point_embedding[labels == 1] += self.point_embeddings[1].weight
  99 |         return point_embedding
 100 | 
 101 |     def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
 102 |         """Embeds box prompts."""
 103 |         boxes = boxes + 0.5  # Shift to center of pixel
 104 |         coords = boxes.reshape(-1, 2, 2)
 105 |         corner_embedding = self.pe_layer.forward_with_coords(
 106 |             coords, self.input_image_size
 107 |         )
 108 |         corner_embedding[:, 0, :] += self.point_embeddings[2].weight
 109 |         corner_embedding[:, 1, :] += self.point_embeddings[3].weight
 110 |         return corner_embedding
 111 | 
 112 |     def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
 113 |         """Embeds mask inputs."""
 114 |         mask_embedding = self.mask_downscaling(masks)
 115 |         return mask_embedding
 116 | 
 117 |     def _get_batch_size(
 118 |         self,
 119 |         points: Optional[Tuple[torch.Tensor, torch.Tensor]],
 120 |         boxes: Optional[torch.Tensor],
 121 |         masks: Optional[torch.Tensor],
 122 |     ) -> int:
 123 |         """
 124 |         Gets the batch size of the output given the batch size of the input prompts.
 125 |         """
 126 |         if points is not None:
 127 |             return points[0].shape[0]
 128 |         elif boxes is not None:
 129 |             return boxes.shape[0]
 130 |         elif masks is not None:
 131 |             return masks.shape[0]
 132 |         else:
 133 |             return 1
 134 | 
 135 |     def _get_device(self) -> torch.device:
 136 |         return self.point_embeddings[0].weight.device
 137 | 
 138 |     def forward(
 139 |         self,
 140 |         points: Optional[Tuple[torch.Tensor, torch.Tensor]],
 141 |         boxes: Optional[torch.Tensor],
 142 |         masks: Optional[torch.Tensor],
 143 |     ) -> Tuple[torch.Tensor, torch.Tensor]:
 144 |         """
 145 |         Embeds different types of prompts, returning both sparse and dense
 146 |         embeddings.
 147 | 
 148 |         Arguments:
 149 |           points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
 150 |             and labels to embed.
 151 |           boxes (torch.Tensor or none): boxes to embed
 152 |           masks (torch.Tensor or none): masks to embed
 153 | 
 154 |         Returns:
 155 |           torch.Tensor: sparse embeddings for the points and boxes, with shape
 156 |             BxNx(embed_dim), where N is determined by the number of input points
 157 |             and boxes.
 158 |           torch.Tensor: dense embeddings for the masks, in the shape
 159 |             Bx(embed_dim)x(embed_H)x(embed_W)
 160 |         """
 161 |         bs = self._get_batch_size(points, boxes, masks)
 162 |         sparse_embeddings = torch.empty(
 163 |             (bs, 0, self.embed_dim), device=self._get_device()
 164 |         )
 165 |         if points is not None:
 166 |             coords, labels = points
 167 |             point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
 168 |             sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
 169 |         if boxes is not None:
 170 |             box_embeddings = self._embed_boxes(boxes)
 171 |             sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
 172 | 
 173 |         if masks is not None:
 174 |             dense_embeddings = self._embed_masks(masks)
 175 |         else:
 176 |             dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
 177 |                 bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
 178 |             )
 179 | 
 180 |         return sparse_embeddings, dense_embeddings
 181 | 
 182 | 
 183 | class PositionEmbeddingRandom(nn.Module):
 184 |     """
 185 |     Positional encoding using random spatial frequencies.
 186 |     """
 187 | 
 188 |     def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
 189 |         super().__init__()
 190 |         if scale is None or scale <= 0.0:
 191 |             scale = 1.0
 192 |         self.register_buffer(
 193 |             "positional_encoding_gaussian_matrix",
 194 |             scale * torch.randn((2, num_pos_feats)),
 195 |         )
 196 | 
 197 |     def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
 198 |         """Positionally encode points that are normalized to [0,1]."""
 199 |         # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
 200 |         coords = 2 * coords - 1
 201 |         coords = coords @ self.positional_encoding_gaussian_matrix
 202 |         coords = 2 * np.pi * coords
 203 |         # outputs d_1 x ... x d_n x C shape
 204 |         return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
 205 | 
 206 |     def forward(self, size: Tuple[int, int]) -> torch.Tensor:
 207 |         """Generate positional encoding for a grid of the specified size."""
 208 |         h, w = size
 209 |         device: Any = self.positional_encoding_gaussian_matrix.device
 210 |         grid = torch.ones((h, w), device=device, dtype=torch.float32)
 211 |         y_embed = grid.cumsum(dim=0) - 0.5
 212 |         x_embed = grid.cumsum(dim=1) - 0.5
 213 |         y_embed = y_embed / h
 214 |         x_embed = x_embed / w
 215 | 
 216 |         pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
 217 |         return pe.permute(2, 0, 1)  # C x H x W
 218 | 
 219 |     def forward_with_coords(
 220 |         self, coords_input: torch.Tensor, image_size: Tuple[int, int]
 221 |     ) -> torch.Tensor:
 222 |         """Positionally encode points that are not normalized to [0,1]."""
 223 |         coords = coords_input.clone()
 224 |         coords[:, :, 0] = coords[:, :, 0] / image_size[1]
 225 |         coords[:, :, 1] = coords[:, :, 1] / image_size[0]
 226 |         return self._pe_encoding(coords.to(torch.float))  # B x N x C
```
