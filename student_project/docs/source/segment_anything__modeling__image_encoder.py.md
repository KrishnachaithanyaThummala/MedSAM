# segment_anything/modeling/image_encoder.py

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
  10 | import torch.nn.functional as F
  11 | 
  12 | from typing import Optional, Tuple, Type
  13 | 
  14 | from .common import LayerNorm2d, MLPBlock
  15 | 
  16 | 
  17 | # This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
  18 | class ImageEncoderViT(nn.Module):
  19 |     def __init__(
  20 |         self,
  21 |         img_size: int = 1024,
  22 |         patch_size: int = 16,
  23 |         in_chans: int = 3,
  24 |         embed_dim: int = 768,
  25 |         depth: int = 12,
  26 |         num_heads: int = 12,
  27 |         mlp_ratio: float = 4.0,
  28 |         out_chans: int = 256,
  29 |         qkv_bias: bool = True,
  30 |         norm_layer: Type[nn.Module] = nn.LayerNorm,
  31 |         act_layer: Type[nn.Module] = nn.GELU,
  32 |         use_abs_pos: bool = True,
  33 |         use_rel_pos: bool = False,
  34 |         rel_pos_zero_init: bool = True,
  35 |         window_size: int = 0,
  36 |         global_attn_indexes: Tuple[int, ...] = (),
  37 |     ) -> None:
  38 |         """
  39 |         Args:
  40 |             img_size (int): Input image size.
  41 |             patch_size (int): Patch size.
  42 |             in_chans (int): Number of input image channels.
  43 |             embed_dim (int): Patch embedding dimension.
  44 |             depth (int): Depth of ViT.
  45 |             num_heads (int): Number of attention heads in each ViT block.
  46 |             mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
  47 |             qkv_bias (bool): If True, add a learnable bias to query, key, value.
  48 |             norm_layer (nn.Module): Normalization layer.
  49 |             act_layer (nn.Module): Activation layer.
  50 |             use_abs_pos (bool): If True, use absolute positional embeddings.
  51 |             use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
  52 |             rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
  53 |             window_size (int): Window size for window attention blocks.
  54 |             global_attn_indexes (list): Indexes for blocks using global attention.
  55 |         """
  56 |         super().__init__()
  57 |         self.img_size = img_size
  58 | 
  59 |         self.patch_embed = PatchEmbed(
  60 |             kernel_size=(patch_size, patch_size),
  61 |             stride=(patch_size, patch_size),
  62 |             in_chans=in_chans,
  63 |             embed_dim=embed_dim,
  64 |         )
  65 | 
  66 |         self.pos_embed: Optional[nn.Parameter] = None
  67 |         if use_abs_pos:
  68 |             # Initialize absolute positional embedding with pretrain image size.
  69 |             self.pos_embed = nn.Parameter(
  70 |                 torch.zeros(
  71 |                     1, img_size // patch_size, img_size // patch_size, embed_dim
  72 |                 )
  73 |             )
  74 | 
  75 |         self.blocks = nn.ModuleList()
  76 |         for i in range(depth):
  77 |             block = Block(
  78 |                 dim=embed_dim,
  79 |                 num_heads=num_heads,
  80 |                 mlp_ratio=mlp_ratio,
  81 |                 qkv_bias=qkv_bias,
  82 |                 norm_layer=norm_layer,
  83 |                 act_layer=act_layer,
  84 |                 use_rel_pos=use_rel_pos,
  85 |                 rel_pos_zero_init=rel_pos_zero_init,
  86 |                 window_size=window_size if i not in global_attn_indexes else 0,
  87 |                 input_size=(img_size // patch_size, img_size // patch_size),
  88 |             )
  89 |             self.blocks.append(block)
  90 | 
  91 |         self.neck = nn.Sequential(
  92 |             nn.Conv2d(
  93 |                 embed_dim,
  94 |                 out_chans,
  95 |                 kernel_size=1,
  96 |                 bias=False,
  97 |             ),
  98 |             LayerNorm2d(out_chans),
  99 |             nn.Conv2d(
 100 |                 out_chans,
 101 |                 out_chans,
 102 |                 kernel_size=3,
 103 |                 padding=1,
 104 |                 bias=False,
 105 |             ),
 106 |             LayerNorm2d(out_chans),
 107 |         )
 108 | 
 109 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
 110 |         x = self.patch_embed(x)
 111 |         if self.pos_embed is not None:
 112 |             x = x + self.pos_embed
 113 | 
 114 |         for blk in self.blocks:
 115 |             x = blk(x)
 116 | 
 117 |         x = self.neck(x.permute(0, 3, 1, 2))
 118 | 
 119 |         return x
 120 | 
 121 | 
 122 | class Block(nn.Module):
 123 |     """Transformer blocks with support of window attention and residual propagation blocks"""
 124 | 
 125 |     def __init__(
 126 |         self,
 127 |         dim: int,
 128 |         num_heads: int,
 129 |         mlp_ratio: float = 4.0,
 130 |         qkv_bias: bool = True,
 131 |         norm_layer: Type[nn.Module] = nn.LayerNorm,
 132 |         act_layer: Type[nn.Module] = nn.GELU,
 133 |         use_rel_pos: bool = False,
 134 |         rel_pos_zero_init: bool = True,
 135 |         window_size: int = 0,
 136 |         input_size: Optional[Tuple[int, int]] = None,
 137 |     ) -> None:
 138 |         """
 139 |         Args:
 140 |             dim (int): Number of input channels.
 141 |             num_heads (int): Number of attention heads in each ViT block.
 142 |             mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
 143 |             qkv_bias (bool): If True, add a learnable bias to query, key, value.
 144 |             norm_layer (nn.Module): Normalization layer.
 145 |             act_layer (nn.Module): Activation layer.
 146 |             use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
 147 |             rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
 148 |             window_size (int): Window size for window attention blocks. If it equals 0, then
 149 |                 use global attention.
 150 |             input_size (tuple(int, int) or None): Input resolution for calculating the relative
 151 |                 positional parameter size.
 152 |         """
 153 |         super().__init__()
 154 |         self.norm1 = norm_layer(dim)
 155 |         self.attn = Attention(
 156 |             dim,
 157 |             num_heads=num_heads,
 158 |             qkv_bias=qkv_bias,
 159 |             use_rel_pos=use_rel_pos,
 160 |             rel_pos_zero_init=rel_pos_zero_init,
 161 |             input_size=input_size if window_size == 0 else (window_size, window_size),
 162 |         )
 163 | 
 164 |         self.norm2 = norm_layer(dim)
 165 |         self.mlp = MLPBlock(
 166 |             embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer
 167 |         )
 168 | 
 169 |         self.window_size = window_size
 170 | 
 171 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
 172 |         shortcut = x
 173 |         x = self.norm1(x)
 174 |         # Window partition
 175 |         if self.window_size > 0:
 176 |             H, W = x.shape[1], x.shape[2]
 177 |             x, pad_hw = window_partition(x, self.window_size)
 178 | 
 179 |         x = self.attn(x)
 180 |         # Reverse window partition
 181 |         if self.window_size > 0:
 182 |             x = window_unpartition(x, self.window_size, pad_hw, (H, W))
 183 | 
 184 |         x = shortcut + x
 185 |         x = x + self.mlp(self.norm2(x))
 186 | 
 187 |         return x
 188 | 
 189 | 
 190 | class Attention(nn.Module):
 191 |     """Multi-head Attention block with relative position embeddings."""
 192 | 
 193 |     def __init__(
 194 |         self,
 195 |         dim: int,
 196 |         num_heads: int = 8,
 197 |         qkv_bias: bool = True,
 198 |         use_rel_pos: bool = False,
 199 |         rel_pos_zero_init: bool = True,
 200 |         input_size: Optional[Tuple[int, int]] = None,
 201 |     ) -> None:
 202 |         """
 203 |         Args:
 204 |             dim (int): Number of input channels.
 205 |             num_heads (int): Number of attention heads.
 206 |             qkv_bias (bool):  If True, add a learnable bias to query, key, value.
 207 |             rel_pos (bool): If True, add relative positional embeddings to the attention map.
 208 |             rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
 209 |             input_size (tuple(int, int) or None): Input resolution for calculating the relative
 210 |                 positional parameter size.
 211 |         """
 212 |         super().__init__()
 213 |         self.num_heads = num_heads
 214 |         head_dim = dim // num_heads
 215 |         self.scale = head_dim**-0.5
 216 | 
 217 |         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
 218 |         self.proj = nn.Linear(dim, dim)
 219 | 
 220 |         self.use_rel_pos = use_rel_pos
 221 |         if self.use_rel_pos:
 222 |             assert (
 223 |                 input_size is not None
 224 |             ), "Input size must be provided if using relative positional encoding."
 225 |             # initialize relative positional embeddings
 226 |             self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
 227 |             self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
 228 | 
 229 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
 230 |         B, H, W, _ = x.shape
 231 |         # qkv with shape (3, B, nHead, H * W, C)
 232 |         qkv = (
 233 |             self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
 234 |         )
 235 |         # q, k, v with shape (B * nHead, H * W, C)
 236 |         q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
 237 | 
 238 |         attn = (q * self.scale) @ k.transpose(-2, -1)
 239 | 
 240 |         if self.use_rel_pos:
 241 |             attn = add_decomposed_rel_pos(
 242 |                 attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
 243 |             )
 244 | 
 245 |         attn = attn.softmax(dim=-1)
 246 |         x = (
 247 |             (attn @ v)
 248 |             .view(B, self.num_heads, H, W, -1)
 249 |             .permute(0, 2, 3, 1, 4)
 250 |             .reshape(B, H, W, -1)
 251 |         )
 252 |         x = self.proj(x)
 253 | 
 254 |         return x
 255 | 
 256 | 
 257 | def window_partition(
 258 |     x: torch.Tensor, window_size: int
 259 | ) -> Tuple[torch.Tensor, Tuple[int, int]]:
 260 |     """
 261 |     Partition into non-overlapping windows with padding if needed.
 262 |     Args:
 263 |         x (tensor): input tokens with [B, H, W, C].
 264 |         window_size (int): window size.
 265 | 
 266 |     Returns:
 267 |         windows: windows after partition with [B * num_windows, window_size, window_size, C].
 268 |         (Hp, Wp): padded height and width before partition
 269 |     """
 270 |     B, H, W, C = x.shape
 271 | 
 272 |     pad_h = (window_size - H % window_size) % window_size
 273 |     pad_w = (window_size - W % window_size) % window_size
 274 |     if pad_h > 0 or pad_w > 0:
 275 |         x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
 276 |     Hp, Wp = H + pad_h, W + pad_w
 277 | 
 278 |     x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
 279 |     windows = (
 280 |         x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
 281 |     )
 282 |     return windows, (Hp, Wp)
 283 | 
 284 | 
 285 | def window_unpartition(
 286 |     windows: torch.Tensor,
 287 |     window_size: int,
 288 |     pad_hw: Tuple[int, int],
 289 |     hw: Tuple[int, int],
 290 | ) -> torch.Tensor:
 291 |     """
 292 |     Window unpartition into original sequences and removing padding.
 293 |     Args:
 294 |         windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
 295 |         window_size (int): window size.
 296 |         pad_hw (Tuple): padded height and width (Hp, Wp).
 297 |         hw (Tuple): original height and width (H, W) before padding.
 298 | 
 299 |     Returns:
 300 |         x: unpartitioned sequences with [B, H, W, C].
 301 |     """
 302 |     Hp, Wp = pad_hw
 303 |     H, W = hw
 304 |     B = windows.shape[0] // (Hp * Wp // window_size // window_size)
 305 |     x = windows.view(
 306 |         B, Hp // window_size, Wp // window_size, window_size, window_size, -1
 307 |     )
 308 |     x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
 309 | 
 310 |     if Hp > H or Wp > W:
 311 |         x = x[:, :H, :W, :].contiguous()
 312 |     return x
 313 | 
 314 | 
 315 | def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
 316 |     """
 317 |     Get relative positional embeddings according to the relative positions of
 318 |         query and key sizes.
 319 |     Args:
 320 |         q_size (int): size of query q.
 321 |         k_size (int): size of key k.
 322 |         rel_pos (Tensor): relative position embeddings (L, C).
 323 | 
 324 |     Returns:
 325 |         Extracted positional embeddings according to relative positions.
 326 |     """
 327 |     max_rel_dist = int(2 * max(q_size, k_size) - 1)
 328 |     # Interpolate rel pos if needed.
 329 |     if rel_pos.shape[0] != max_rel_dist:
 330 |         # Interpolate rel pos.
 331 |         rel_pos_resized = F.interpolate(
 332 |             rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
 333 |             size=max_rel_dist,
 334 |             mode="linear",
 335 |         )
 336 |         rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
 337 |     else:
 338 |         rel_pos_resized = rel_pos
 339 | 
 340 |     # Scale the coords with short length if shapes for q and k are different.
 341 |     q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
 342 |     k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
 343 |     relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
 344 | 
 345 |     return rel_pos_resized[relative_coords.long()]
 346 | 
 347 | 
 348 | def add_decomposed_rel_pos(
 349 |     attn: torch.Tensor,
 350 |     q: torch.Tensor,
 351 |     rel_pos_h: torch.Tensor,
 352 |     rel_pos_w: torch.Tensor,
 353 |     q_size: Tuple[int, int],
 354 |     k_size: Tuple[int, int],
 355 | ) -> torch.Tensor:
 356 |     """
 357 |     Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
 358 |     https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
 359 |     Args:
 360 |         attn (Tensor): attention map.
 361 |         q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
 362 |         rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
 363 |         rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
 364 |         q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
 365 |         k_size (Tuple): spatial sequence size of key k with (k_h, k_w).
 366 | 
 367 |     Returns:
 368 |         attn (Tensor): attention map with added relative positional embeddings.
 369 |     """
 370 |     q_h, q_w = q_size
 371 |     k_h, k_w = k_size
 372 |     Rh = get_rel_pos(q_h, k_h, rel_pos_h)
 373 |     Rw = get_rel_pos(q_w, k_w, rel_pos_w)
 374 | 
 375 |     B, _, dim = q.shape
 376 |     r_q = q.reshape(B, q_h, q_w, dim)
 377 |     rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
 378 |     rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
 379 | 
 380 |     attn = (
 381 |         attn.view(B, q_h, q_w, k_h, k_w)
 382 |         + rel_h[:, :, :, :, None]
 383 |         + rel_w[:, :, :, None, :]
 384 |     ).view(B, q_h * q_w, k_h * k_w)
 385 | 
 386 |     return attn
 387 | 
 388 | 
 389 | class PatchEmbed(nn.Module):
 390 |     """
 391 |     Image to Patch Embedding.
 392 |     """
 393 | 
 394 |     def __init__(
 395 |         self,
 396 |         kernel_size: Tuple[int, int] = (16, 16),
 397 |         stride: Tuple[int, int] = (16, 16),
 398 |         padding: Tuple[int, int] = (0, 0),
 399 |         in_chans: int = 3,
 400 |         embed_dim: int = 768,
 401 |     ) -> None:
 402 |         """
 403 |         Args:
 404 |             kernel_size (Tuple): kernel size of the projection layer.
 405 |             stride (Tuple): stride of the projection layer.
 406 |             padding (Tuple): padding size of the projection layer.
 407 |             in_chans (int): Number of input image channels.
 408 |             embed_dim (int): Patch embedding dimension.
 409 |         """
 410 |         super().__init__()
 411 | 
 412 |         self.proj = nn.Conv2d(
 413 |             in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
 414 |         )
 415 | 
 416 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
 417 |         x = self.proj(x)
 418 |         # B C H W -> B H W C
 419 |         x = x.permute(0, 2, 3, 1)
 420 |         return x
```
