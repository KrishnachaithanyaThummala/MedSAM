# segment_anything/modeling/mask_decoder.py

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
  12 | from typing import List, Tuple, Type
  13 | 
  14 | from .common import LayerNorm2d
  15 | 
  16 | 
  17 | class MaskDecoder(nn.Module):
  18 |     def __init__(
  19 |         self,
  20 |         *,
  21 |         transformer_dim: int,
  22 |         transformer: nn.Module,
  23 |         num_multimask_outputs: int = 3,
  24 |         activation: Type[nn.Module] = nn.GELU,
  25 |         iou_head_depth: int = 3,
  26 |         iou_head_hidden_dim: int = 256,
  27 |     ) -> None:
  28 |         """
  29 |         Predicts masks given an image and prompt embeddings, using a
  30 |         transformer architecture.
  31 | 
  32 |         Arguments:
  33 |           transformer_dim (int): the channel dimension of the transformer
  34 |           transformer (nn.Module): the transformer used to predict masks
  35 |           num_multimask_outputs (int): the number of masks to predict
  36 |             when disambiguating masks
  37 |           activation (nn.Module): the type of activation to use when
  38 |             upscaling masks
  39 |           iou_head_depth (int): the depth of the MLP used to predict
  40 |             mask quality
  41 |           iou_head_hidden_dim (int): the hidden dimension of the MLP
  42 |             used to predict mask quality
  43 |         """
  44 |         super().__init__()
  45 |         self.transformer_dim = transformer_dim
  46 |         self.transformer = transformer
  47 | 
  48 |         self.num_multimask_outputs = num_multimask_outputs
  49 | 
  50 |         self.iou_token = nn.Embedding(1, transformer_dim)
  51 |         self.num_mask_tokens = num_multimask_outputs + 1
  52 |         self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
  53 | 
  54 |         self.output_upscaling = nn.Sequential(
  55 |             nn.ConvTranspose2d(
  56 |                 transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
  57 |             ),
  58 |             LayerNorm2d(transformer_dim // 4),
  59 |             activation(),
  60 |             nn.ConvTranspose2d(
  61 |                 transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
  62 |             ),
  63 |             activation(),
  64 |         )
  65 |         self.output_hypernetworks_mlps = nn.ModuleList(
  66 |             [
  67 |                 MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
  68 |                 for i in range(self.num_mask_tokens)
  69 |             ]
  70 |         )
  71 | 
  72 |         self.iou_prediction_head = MLP(
  73 |             transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
  74 |         )
  75 | 
  76 |     def forward(
  77 |         self,
  78 |         image_embeddings: torch.Tensor,
  79 |         image_pe: torch.Tensor,
  80 |         sparse_prompt_embeddings: torch.Tensor,
  81 |         dense_prompt_embeddings: torch.Tensor,
  82 |         multimask_output: bool,
  83 |     ) -> Tuple[torch.Tensor, torch.Tensor]:
  84 |         """
  85 |         Predict masks given image and prompt embeddings.
  86 | 
  87 |         Arguments:
  88 |           image_embeddings (torch.Tensor): the embeddings from the image encoder
  89 |           image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
  90 |           sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
  91 |           dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
  92 |           multimask_output (bool): Whether to return multiple masks or a single
  93 |             mask.
  94 | 
  95 |         Returns:
  96 |           torch.Tensor: batched predicted masks
  97 |           torch.Tensor: batched predictions of mask quality
  98 |         """
  99 |         masks, iou_pred = self.predict_masks(
 100 |             image_embeddings=image_embeddings,
 101 |             image_pe=image_pe,
 102 |             sparse_prompt_embeddings=sparse_prompt_embeddings,
 103 |             dense_prompt_embeddings=dense_prompt_embeddings,
 104 |         )
 105 | 
 106 |         # Select the correct mask or masks for output
 107 |         if multimask_output:
 108 |             mask_slice = slice(1, None)
 109 |         else:
 110 |             mask_slice = slice(0, 1)
 111 |         masks = masks[:, mask_slice, :, :]
 112 |         iou_pred = iou_pred[:, mask_slice]
 113 | 
 114 |         # Prepare output
 115 |         return masks, iou_pred
 116 | 
 117 |     def predict_masks(
 118 |         self,
 119 |         image_embeddings: torch.Tensor,
 120 |         image_pe: torch.Tensor,
 121 |         sparse_prompt_embeddings: torch.Tensor,
 122 |         dense_prompt_embeddings: torch.Tensor,
 123 |     ) -> Tuple[torch.Tensor, torch.Tensor]:
 124 |         """Predicts masks. See 'forward' for more details."""
 125 |         # Concatenate output tokens
 126 |         output_tokens = torch.cat(
 127 |             [self.iou_token.weight, self.mask_tokens.weight], dim=0
 128 |         )
 129 |         output_tokens = output_tokens.unsqueeze(0).expand(
 130 |             sparse_prompt_embeddings.size(0), -1, -1
 131 |         )
 132 |         tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
 133 | 
 134 |         # Expand per-image data in batch direction to be per-mask
 135 |         if image_embeddings.shape[0] != tokens.shape[0]:
 136 |             src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
 137 |         else:
 138 |             src = image_embeddings
 139 |         src = src + dense_prompt_embeddings
 140 |         pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
 141 |         b, c, h, w = src.shape
 142 | 
 143 |         # Run the transformer
 144 |         hs, src = self.transformer(src, pos_src, tokens)
 145 |         iou_token_out = hs[:, 0, :]
 146 |         mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]
 147 | 
 148 |         # Upscale mask embeddings and predict masks using the mask tokens
 149 |         src = src.transpose(1, 2).view(b, c, h, w)
 150 |         upscaled_embedding = self.output_upscaling(src)
 151 |         hyper_in_list: List[torch.Tensor] = []
 152 |         for i in range(self.num_mask_tokens):
 153 |             hyper_in_list.append(
 154 |                 self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
 155 |             )
 156 |         hyper_in = torch.stack(hyper_in_list, dim=1)
 157 |         b, c, h, w = upscaled_embedding.shape
 158 |         masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
 159 | 
 160 |         # Generate mask quality predictions
 161 |         iou_pred = self.iou_prediction_head(iou_token_out)
 162 | 
 163 |         return masks, iou_pred
 164 | 
 165 | 
 166 | # Lightly adapted from
 167 | # https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
 168 | class MLP(nn.Module):
 169 |     def __init__(
 170 |         self,
 171 |         input_dim: int,
 172 |         hidden_dim: int,
 173 |         output_dim: int,
 174 |         num_layers: int,
 175 |         sigmoid_output: bool = False,
 176 |     ) -> None:
 177 |         super().__init__()
 178 |         self.num_layers = num_layers
 179 |         h = [hidden_dim] * (num_layers - 1)
 180 |         self.layers = nn.ModuleList(
 181 |             nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
 182 |         )
 183 |         self.sigmoid_output = sigmoid_output
 184 | 
 185 |     def forward(self, x):
 186 |         for i, layer in enumerate(self.layers):
 187 |             x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
 188 |         if self.sigmoid_output:
 189 |             x = F.sigmoid(x)
 190 |         return x
```
