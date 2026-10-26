# segment_anything/modeling/transformer.py

```python
   1 | # -*- coding: utf-8 -*-
   2 | # Copyright (c) Meta Platforms, Inc. and affiliates.
   3 | # All rights reserved.
   4 | 
   5 | # This source code is licensed under the license found in the
   6 | # LICENSE file in the root directory of this source tree.
   7 | 
   8 | import torch
   9 | from torch import Tensor, nn
  10 | 
  11 | import math
  12 | from typing import Tuple, Type
  13 | 
  14 | from .common import MLPBlock
  15 | 
  16 | 
  17 | class TwoWayTransformer(nn.Module):
  18 |     def __init__(
  19 |         self,
  20 |         depth: int,
  21 |         embedding_dim: int,
  22 |         num_heads: int,
  23 |         mlp_dim: int,
  24 |         activation: Type[nn.Module] = nn.ReLU,
  25 |         attention_downsample_rate: int = 2,
  26 |     ) -> None:
  27 |         """
  28 |         A transformer decoder that attends to an input image using
  29 |         queries whose positional embedding is supplied.
  30 | 
  31 |         Args:
  32 |           depth (int): number of layers in the transformer
  33 |           embedding_dim (int): the channel dimension for the input embeddings
  34 |           num_heads (int): the number of heads for multihead attention. Must
  35 |             divide embedding_dim
  36 |           mlp_dim (int): the channel dimension internal to the MLP block
  37 |           activation (nn.Module): the activation to use in the MLP block
  38 |         """
  39 |         super().__init__()
  40 |         self.depth = depth
  41 |         self.embedding_dim = embedding_dim
  42 |         self.num_heads = num_heads
  43 |         self.mlp_dim = mlp_dim
  44 |         self.layers = nn.ModuleList()
  45 | 
  46 |         for i in range(depth):
  47 |             self.layers.append(
  48 |                 TwoWayAttentionBlock(
  49 |                     embedding_dim=embedding_dim,
  50 |                     num_heads=num_heads,
  51 |                     mlp_dim=mlp_dim,
  52 |                     activation=activation,
  53 |                     attention_downsample_rate=attention_downsample_rate,
  54 |                     skip_first_layer_pe=(i == 0),
  55 |                 )
  56 |             )
  57 | 
  58 |         self.final_attn_token_to_image = Attention(
  59 |             embedding_dim, num_heads, downsample_rate=attention_downsample_rate
  60 |         )
  61 |         self.norm_final_attn = nn.LayerNorm(embedding_dim)
  62 | 
  63 |     def forward(
  64 |         self,
  65 |         image_embedding: Tensor,
  66 |         image_pe: Tensor,
  67 |         point_embedding: Tensor,
  68 |     ) -> Tuple[Tensor, Tensor]:
  69 |         """
  70 |         Args:
  71 |           image_embedding (torch.Tensor): image to attend to. Should be shape
  72 |             B x embedding_dim x h x w for any h and w.
  73 |           image_pe (torch.Tensor): the positional encoding to add to the image. Must
  74 |             have the same shape as image_embedding.
  75 |           point_embedding (torch.Tensor): the embedding to add to the query points.
  76 |             Must have shape B x N_points x embedding_dim for any N_points.
  77 | 
  78 |         Returns:
  79 |           torch.Tensor: the processed point_embedding
  80 |           torch.Tensor: the processed image_embedding
  81 |         """
  82 |         # BxCxHxW -> BxHWxC == B x N_image_tokens x C
  83 |         bs, c, h, w = image_embedding.shape
  84 |         image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
  85 |         image_pe = image_pe.flatten(2).permute(0, 2, 1)
  86 | 
  87 |         # Prepare queries
  88 |         queries = point_embedding
  89 |         keys = image_embedding
  90 | 
  91 |         # Apply transformer blocks and final layernorm
  92 |         for layer in self.layers:
  93 |             queries, keys = layer(
  94 |                 queries=queries,
  95 |                 keys=keys,
  96 |                 query_pe=point_embedding,
  97 |                 key_pe=image_pe,
  98 |             )
  99 | 
 100 |         # Apply the final attention layer from the points to the image
 101 |         q = queries + point_embedding
 102 |         k = keys + image_pe
 103 |         attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
 104 |         queries = queries + attn_out
 105 |         queries = self.norm_final_attn(queries)
 106 | 
 107 |         return queries, keys
 108 | 
 109 | 
 110 | class TwoWayAttentionBlock(nn.Module):
 111 |     def __init__(
 112 |         self,
 113 |         embedding_dim: int,
 114 |         num_heads: int,
 115 |         mlp_dim: int = 2048,
 116 |         activation: Type[nn.Module] = nn.ReLU,
 117 |         attention_downsample_rate: int = 2,
 118 |         skip_first_layer_pe: bool = False,
 119 |     ) -> None:
 120 |         """
 121 |         A transformer block with four layers: (1) self-attention of sparse
 122 |         inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
 123 |         block on sparse inputs, and (4) cross attention of dense inputs to sparse
 124 |         inputs.
 125 | 
 126 |         Arguments:
 127 |           embedding_dim (int): the channel dimension of the embeddings
 128 |           num_heads (int): the number of heads in the attention layers
 129 |           mlp_dim (int): the hidden dimension of the mlp block
 130 |           activation (nn.Module): the activation of the mlp block
 131 |           skip_first_layer_pe (bool): skip the PE on the first layer
 132 |         """
 133 |         super().__init__()
 134 |         self.self_attn = Attention(embedding_dim, num_heads)
 135 |         self.norm1 = nn.LayerNorm(embedding_dim)
 136 | 
 137 |         self.cross_attn_token_to_image = Attention(
 138 |             embedding_dim, num_heads, downsample_rate=attention_downsample_rate
 139 |         )
 140 |         self.norm2 = nn.LayerNorm(embedding_dim)
 141 | 
 142 |         self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
 143 |         self.norm3 = nn.LayerNorm(embedding_dim)
 144 | 
 145 |         self.norm4 = nn.LayerNorm(embedding_dim)
 146 |         self.cross_attn_image_to_token = Attention(
 147 |             embedding_dim, num_heads, downsample_rate=attention_downsample_rate
 148 |         )
 149 | 
 150 |         self.skip_first_layer_pe = skip_first_layer_pe
 151 | 
 152 |     def forward(
 153 |         self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
 154 |     ) -> Tuple[Tensor, Tensor]:
 155 |         # Self attention block
 156 |         if self.skip_first_layer_pe:
 157 |             queries = self.self_attn(q=queries, k=queries, v=queries)
 158 |         else:
 159 |             q = queries + query_pe
 160 |             attn_out = self.self_attn(q=q, k=q, v=queries)
 161 |             queries = queries + attn_out
 162 |         queries = self.norm1(queries)
 163 | 
 164 |         # Cross attention block, tokens attending to image embedding
 165 |         q = queries + query_pe
 166 |         k = keys + key_pe
 167 |         attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
 168 |         queries = queries + attn_out
 169 |         queries = self.norm2(queries)
 170 | 
 171 |         # MLP block
 172 |         mlp_out = self.mlp(queries)
 173 |         queries = queries + mlp_out
 174 |         queries = self.norm3(queries)
 175 | 
 176 |         # Cross attention block, image embedding attending to tokens
 177 |         q = queries + query_pe
 178 |         k = keys + key_pe
 179 |         attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
 180 |         keys = keys + attn_out
 181 |         keys = self.norm4(keys)
 182 | 
 183 |         return queries, keys
 184 | 
 185 | 
 186 | class Attention(nn.Module):
 187 |     """
 188 |     An attention layer that allows for downscaling the size of the embedding
 189 |     after projection to queries, keys, and values.
 190 |     """
 191 | 
 192 |     def __init__(
 193 |         self,
 194 |         embedding_dim: int,
 195 |         num_heads: int,
 196 |         downsample_rate: int = 1,
 197 |     ) -> None:
 198 |         super().__init__()
 199 |         self.embedding_dim = embedding_dim
 200 |         self.internal_dim = embedding_dim // downsample_rate
 201 |         self.num_heads = num_heads
 202 |         assert (
 203 |             self.internal_dim % num_heads == 0
 204 |         ), "num_heads must divide embedding_dim."
 205 | 
 206 |         self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
 207 |         self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
 208 |         self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
 209 |         self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
 210 | 
 211 |     def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
 212 |         b, n, c = x.shape
 213 |         x = x.reshape(b, n, num_heads, c // num_heads)
 214 |         return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head
 215 | 
 216 |     def _recombine_heads(self, x: Tensor) -> Tensor:
 217 |         b, n_heads, n_tokens, c_per_head = x.shape
 218 |         x = x.transpose(1, 2)
 219 |         return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C
 220 | 
 221 |     def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
 222 |         # Input projections
 223 |         q = self.q_proj(q)
 224 |         k = self.k_proj(k)
 225 |         v = self.v_proj(v)
 226 | 
 227 |         # Separate into heads
 228 |         q = self._separate_heads(q, self.num_heads)
 229 |         k = self._separate_heads(k, self.num_heads)
 230 |         v = self._separate_heads(v, self.num_heads)
 231 | 
 232 |         # Attention
 233 |         _, _, _, c_per_head = q.shape
 234 |         attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
 235 |         attn = attn / math.sqrt(c_per_head)
 236 |         attn = torch.softmax(attn, dim=-1)
 237 | 
 238 |         # Get output
 239 |         out = attn @ v
 240 |         out = self._recombine_heads(out)
 241 |         out = self.out_proj(out)
 242 | 
 243 |         return out
```
