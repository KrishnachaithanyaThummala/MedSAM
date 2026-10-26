# distillation/tiny_vit_sam.py

```python
   1 | # --------------------------------------------------------
   2 | # TinyViT Model Architecture
   3 | # Copyright (c) 2022 Microsoft
   4 | # Adapted from LeViT and Swin Transformer
   5 | #   LeViT: (https://github.com/facebookresearch/levit)
   6 | #   Swin: (https://github.com/microsoft/swin-transformer)
   7 | # Build the TinyViT Model
   8 | # --------------------------------------------------------
   9 | 
  10 | import itertools
  11 | import torch
  12 | import torch.nn as nn
  13 | import torch.nn.functional as F
  14 | import torch.utils.checkpoint as checkpoint
  15 | from timm.models.layers import DropPath as TimmDropPath,\
  16 |     to_2tuple, trunc_normal_
  17 | from timm.models.registry import register_model
  18 | from typing import Tuple
  19 | 
  20 | 
  21 | class Conv2d_BN(torch.nn.Sequential):
  22 |     def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
  23 |                  groups=1, bn_weight_init=1):
  24 |         super().__init__()
  25 |         self.add_module('c', torch.nn.Conv2d(
  26 |             a, b, ks, stride, pad, dilation, groups, bias=False))
  27 |         bn = torch.nn.BatchNorm2d(b)
  28 |         torch.nn.init.constant_(bn.weight, bn_weight_init)
  29 |         torch.nn.init.constant_(bn.bias, 0)
  30 |         self.add_module('bn', bn)
  31 | 
  32 |     @torch.no_grad()
  33 |     def fuse(self):
  34 |         c, bn = self._modules.values()
  35 |         w = bn.weight / (bn.running_var + bn.eps)**0.5
  36 |         w = c.weight * w[:, None, None, None]
  37 |         b = bn.bias - bn.running_mean * bn.weight / \
  38 |             (bn.running_var + bn.eps)**0.5
  39 |         m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
  40 |             0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
  41 |         m.weight.data.copy_(w)
  42 |         m.bias.data.copy_(b)
  43 |         return m
  44 | 
  45 | 
  46 | class DropPath(TimmDropPath):
  47 |     def __init__(self, drop_prob=None):
  48 |         super().__init__(drop_prob=drop_prob)
  49 |         self.drop_prob = drop_prob
  50 | 
  51 |     def __repr__(self):
  52 |         msg = super().__repr__()
  53 |         msg += f'(drop_prob={self.drop_prob})'
  54 |         return msg
  55 | 
  56 | 
  57 | class PatchEmbed(nn.Module):
  58 |     def __init__(self, in_chans, embed_dim, resolution, activation):
  59 |         super().__init__()
  60 |         img_size: Tuple[int, int] = to_2tuple(resolution)
  61 |         #self.patches_resolution = (img_size[0] // 4, img_size[1] // 4)
  62 |         self.patches_resolution = img_size
  63 |         self.num_patches = self.patches_resolution[0] * \
  64 |             self.patches_resolution[1]
  65 |         self.in_chans = in_chans
  66 |         self.embed_dim = embed_dim
  67 |         n = embed_dim
  68 |         #self.seq = nn.Sequential(
  69 |         #    Conv2d_BN(in_chans, n // 2, 3, 2, 1),
  70 |         #    activation(),
  71 |         #    Conv2d_BN(n // 2, n, 3, 2, 1),
  72 |         #)
  73 |         self.seq = nn.Sequential(
  74 |             Conv2d_BN(in_chans, n // 2, 1, 1, 0),
  75 |             activation(),
  76 |             Conv2d_BN(n // 2, n, 1, 1, 0),
  77 |         )
  78 | 
  79 |     def forward(self, x):
  80 |         return self.seq(x)
  81 | 
  82 | 
  83 | class MBConv(nn.Module):
  84 |     def __init__(self, in_chans, out_chans, expand_ratio,
  85 |                  activation, drop_path):
  86 |         super().__init__()
  87 |         self.in_chans = in_chans
  88 |         self.hidden_chans = int(in_chans * expand_ratio)
  89 |         self.out_chans = out_chans
  90 | 
  91 |         self.conv1 = Conv2d_BN(in_chans, self.hidden_chans, ks=1)
  92 |         self.act1 = activation()
  93 | 
  94 |         self.conv2 = Conv2d_BN(self.hidden_chans, self.hidden_chans,
  95 |                                ks=3, stride=1, pad=1, groups=self.hidden_chans)
  96 |         self.act2 = activation()
  97 | 
  98 |         self.conv3 = Conv2d_BN(
  99 |             self.hidden_chans, out_chans, ks=1, bn_weight_init=0.0)
 100 |         self.act3 = activation()
 101 | 
 102 |         self.drop_path = DropPath(
 103 |             drop_path) if drop_path > 0. else nn.Identity()
 104 | 
 105 |     def forward(self, x):
 106 |         shortcut = x
 107 | 
 108 |         x = self.conv1(x)
 109 |         x = self.act1(x)
 110 | 
 111 |         x = self.conv2(x)
 112 |         x = self.act2(x)
 113 | 
 114 |         x = self.conv3(x)
 115 | 
 116 |         x = self.drop_path(x)
 117 | 
 118 |         x += shortcut
 119 |         x = self.act3(x)
 120 | 
 121 |         return x
 122 | 
 123 | 
 124 | class PatchMerging(nn.Module):
 125 |     def __init__(self, input_resolution, dim, out_dim, activation):
 126 |         super().__init__()
 127 | 
 128 |         self.input_resolution = input_resolution
 129 |         self.dim = dim
 130 |         self.out_dim = out_dim
 131 |         self.act = activation()
 132 |         self.conv1 = Conv2d_BN(dim, out_dim, 1, 1, 0)
 133 |         stride_c=2
 134 |         if(out_dim==320 or out_dim==448 or out_dim==576):
 135 |             stride_c=1
 136 |         self.conv2 = Conv2d_BN(out_dim, out_dim, 3, stride_c, 1, groups=out_dim)
 137 |         self.conv3 = Conv2d_BN(out_dim, out_dim, 1, 1, 0)
 138 | 
 139 |     def forward(self, x):
 140 |         if x.ndim == 3:
 141 |             H, W = self.input_resolution
 142 |             B = len(x)
 143 |             # (B, C, H, W)
 144 |             x = x.view(B, H, W, -1).permute(0, 3, 1, 2)
 145 | 
 146 |         x = self.conv1(x)
 147 |         x = self.act(x)
 148 | 
 149 |         x = self.conv2(x)
 150 |         x = self.act(x)
 151 |         x = self.conv3(x)
 152 |         x = x.flatten(2).transpose(1, 2)
 153 |         return x
 154 | 
 155 | 
 156 | class ConvLayer(nn.Module):
 157 |     def __init__(self, dim, input_resolution, depth,
 158 |                  activation,
 159 |                  drop_path=0., downsample=None, use_checkpoint=False,
 160 |                  out_dim=None,
 161 |                  conv_expand_ratio=4.,
 162 |                  ):
 163 | 
 164 |         super().__init__()
 165 |         self.dim = dim
 166 |         self.input_resolution = input_resolution
 167 |         self.depth = depth
 168 |         self.use_checkpoint = use_checkpoint
 169 | 
 170 |         # build blocks
 171 |         self.blocks = nn.ModuleList([
 172 |             MBConv(dim, dim, conv_expand_ratio, activation,
 173 |                    drop_path[i] if isinstance(drop_path, list) else drop_path,
 174 |                    )
 175 |             for i in range(depth)])
 176 | 
 177 |         # patch merging layer
 178 |         if downsample is not None:
 179 |             self.downsample = downsample(
 180 |                 input_resolution, dim=dim, out_dim=out_dim, activation=activation)
 181 |         else:
 182 |             self.downsample = None
 183 | 
 184 |     def forward(self, x):
 185 |         for blk in self.blocks:
 186 |             if self.use_checkpoint:
 187 |                 x = checkpoint.checkpoint(blk, x)
 188 |             else:
 189 |                 x = blk(x)
 190 |         if self.downsample is not None:
 191 |             x = self.downsample(x)
 192 |         return x
 193 | 
 194 | 
 195 | class Mlp(nn.Module):
 196 |     def __init__(self, in_features, hidden_features=None,
 197 |                  out_features=None, act_layer=nn.GELU, drop=0.):
 198 |         super().__init__()
 199 |         out_features = out_features or in_features
 200 |         hidden_features = hidden_features or in_features
 201 |         self.norm = nn.LayerNorm(in_features)
 202 |         self.fc1 = nn.Linear(in_features, hidden_features)
 203 |         self.fc2 = nn.Linear(hidden_features, out_features)
 204 |         self.act = act_layer()
 205 |         self.drop = nn.Dropout(drop)
 206 | 
 207 |     def forward(self, x):
 208 |         x = self.norm(x)
 209 | 
 210 |         x = self.fc1(x)
 211 |         x = self.act(x)
 212 |         x = self.drop(x)
 213 |         x = self.fc2(x)
 214 |         x = self.drop(x)
 215 |         return x
 216 | 
 217 | 
 218 | class Attention(torch.nn.Module):
 219 |     def __init__(self, dim, key_dim, num_heads=8,
 220 |                  attn_ratio=4,
 221 |                  resolution=(14, 14),
 222 |                  ):
 223 |         super().__init__()
 224 |         # (h, w)
 225 |         assert isinstance(resolution, tuple) and len(resolution) == 2
 226 |         self.num_heads = num_heads
 227 |         self.scale = key_dim ** -0.5
 228 |         self.key_dim = key_dim
 229 |         self.nh_kd = nh_kd = key_dim * num_heads
 230 |         self.d = int(attn_ratio * key_dim)
 231 |         self.dh = int(attn_ratio * key_dim) * num_heads
 232 |         self.attn_ratio = attn_ratio
 233 |         h = self.dh + nh_kd * 2
 234 | 
 235 |         self.norm = nn.LayerNorm(dim)
 236 |         self.qkv = nn.Linear(dim, h)
 237 |         self.proj = nn.Linear(self.dh, dim)
 238 | 
 239 |         points = list(itertools.product(
 240 |             range(resolution[0]), range(resolution[1])))
 241 |         N = len(points)
 242 |         attention_offsets = {}
 243 |         idxs = []
 244 |         for p1 in points:
 245 |             for p2 in points:
 246 |                 offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
 247 |                 if offset not in attention_offsets:
 248 |                     attention_offsets[offset] = len(attention_offsets)
 249 |                 idxs.append(attention_offsets[offset])
 250 |         self.attention_biases = torch.nn.Parameter(
 251 |             torch.zeros(num_heads, len(attention_offsets)))
 252 |         self.register_buffer('attention_bias_idxs',
 253 |                              torch.LongTensor(idxs).view(N, N),
 254 |                              persistent=False)
 255 | 
 256 |     @torch.no_grad()
 257 |     def train(self, mode=True):
 258 |         super().train(mode)
 259 |         if mode and hasattr(self, 'ab'):
 260 |             del self.ab
 261 |         else:
 262 |             self.register_buffer('ab',
 263 |                                  self.attention_biases[:, self.attention_bias_idxs],
 264 |                                  persistent=False)
 265 | 
 266 |     def forward(self, x):  # x (B,N,C)
 267 |         B, N, _ = x.shape
 268 | 
 269 |         # Normalization
 270 |         x = self.norm(x)
 271 | 
 272 |         qkv = self.qkv(x)
 273 |         # (B, N, num_heads, d)
 274 |         q, k, v = qkv.view(B, N, self.num_heads, -
 275 |                            1).split([self.key_dim, self.key_dim, self.d], dim=3)
 276 |         # (B, num_heads, N, d)
 277 |         q = q.permute(0, 2, 1, 3)
 278 |         k = k.permute(0, 2, 1, 3)
 279 |         v = v.permute(0, 2, 1, 3)
 280 | 
 281 |         attn = (
 282 |             (q @ k.transpose(-2, -1)) * self.scale
 283 |             +
 284 |             (self.attention_biases[:, self.attention_bias_idxs]
 285 |              if self.training else self.ab)
 286 |         )
 287 |         attn = attn.softmax(dim=-1)
 288 |         x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
 289 |         x = self.proj(x)
 290 |         return x
 291 | 
 292 | 
 293 | class TinyViTBlock(nn.Module):
 294 |     r""" TinyViT Block.
 295 | 
 296 |     Args:
 297 |         dim (int): Number of input channels.
 298 |         input_resolution (tuple[int, int]): Input resolution.
 299 |         num_heads (int): Number of attention heads.
 300 |         window_size (int): Window size.
 301 |         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
 302 |         drop (float, optional): Dropout rate. Default: 0.0
 303 |         drop_path (float, optional): Stochastic depth rate. Default: 0.0
 304 |         local_conv_size (int): the kernel size of the convolution between
 305 |                                Attention and MLP. Default: 3
 306 |         activation: the activation function. Default: nn.GELU
 307 |     """
 308 | 
 309 |     def __init__(self, dim, input_resolution, num_heads, window_size=7,
 310 |                  mlp_ratio=4., drop=0., drop_path=0.,
 311 |                  local_conv_size=3,
 312 |                  activation=nn.GELU,
 313 |                  ):
 314 |         super().__init__()
 315 |         self.dim = dim
 316 |         self.input_resolution = input_resolution
 317 |         self.num_heads = num_heads
 318 |         assert window_size > 0, 'window_size must be greater than 0'
 319 |         self.window_size = window_size
 320 |         self.mlp_ratio = mlp_ratio
 321 | 
 322 |         self.drop_path = DropPath(
 323 |             drop_path) if drop_path > 0. else nn.Identity()
 324 | 
 325 |         assert dim % num_heads == 0, 'dim must be divisible by num_heads'
 326 |         head_dim = dim // num_heads
 327 | 
 328 |         window_resolution = (window_size, window_size)
 329 |         self.attn = Attention(dim, head_dim, num_heads,
 330 |                               attn_ratio=1, resolution=window_resolution)
 331 | 
 332 |         mlp_hidden_dim = int(dim * mlp_ratio)
 333 |         mlp_activation = activation
 334 |         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
 335 |                        act_layer=mlp_activation, drop=drop)
 336 | 
 337 |         pad = local_conv_size // 2
 338 |         self.local_conv = Conv2d_BN(
 339 |             dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim)
 340 | 
 341 |     def forward(self, x):
 342 |         H, W = self.input_resolution
 343 |         B, L, C = x.shape
 344 |         assert L == H * W, "input feature has wrong size"
 345 |         res_x = x
 346 |         if H == self.window_size and W == self.window_size:
 347 |             x = self.attn(x)
 348 |         else:
 349 |             x = x.view(B, H, W, C)
 350 |             pad_b = (self.window_size - H %
 351 |                      self.window_size) % self.window_size
 352 |             pad_r = (self.window_size - W %
 353 |                      self.window_size) % self.window_size
 354 |             padding = pad_b > 0 or pad_r > 0
 355 | 
 356 |             if padding:
 357 |                 x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
 358 | 
 359 |             pH, pW = H + pad_b, W + pad_r
 360 |             nH = pH // self.window_size
 361 |             nW = pW // self.window_size
 362 |             # window partition
 363 |             x = x.view(B, nH, self.window_size, nW, self.window_size, C).transpose(2, 3).reshape(
 364 |                 B * nH * nW, self.window_size * self.window_size, C)
 365 |             x = self.attn(x)
 366 |             # window reverse
 367 |             x = x.view(B, nH, nW, self.window_size, self.window_size,
 368 |                        C).transpose(2, 3).reshape(B, pH, pW, C)
 369 | 
 370 |             if padding:
 371 |                 x = x[:, :H, :W].contiguous()
 372 | 
 373 |             x = x.view(B, L, C)
 374 | 
 375 |         x = res_x + self.drop_path(x)
 376 | 
 377 |         x = x.transpose(1, 2).reshape(B, C, H, W)
 378 |         x = self.local_conv(x)
 379 |         x = x.view(B, C, L).transpose(1, 2)
 380 | 
 381 |         x = x + self.drop_path(self.mlp(x))
 382 |         return x
 383 | 
 384 |     def extra_repr(self) -> str:
 385 |         return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
 386 |                f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"
 387 | 
 388 | 
 389 | class BasicLayer(nn.Module):
 390 |     """ A basic TinyViT layer for one stage.
 391 | 
 392 |     Args:
 393 |         dim (int): Number of input channels.
 394 |         input_resolution (tuple[int]): Input resolution.
 395 |         depth (int): Number of blocks.
 396 |         num_heads (int): Number of attention heads.
 397 |         window_size (int): Local window size.
 398 |         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
 399 |         drop (float, optional): Dropout rate. Default: 0.0
 400 |         drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
 401 |         downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
 402 |         use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
 403 |         local_conv_size: the kernel size of the depthwise convolution between attention and MLP. Default: 3
 404 |         activation: the activation function. Default: nn.GELU
 405 |         out_dim: the output dimension of the layer. Default: dim
 406 |     """
 407 | 
 408 |     def __init__(self, dim, input_resolution, depth, num_heads, window_size,
 409 |                  mlp_ratio=4., drop=0.,
 410 |                  drop_path=0., downsample=None, use_checkpoint=False,
 411 |                  local_conv_size=3,
 412 |                  activation=nn.GELU,
 413 |                  out_dim=None,
 414 |                  ):
 415 | 
 416 |         super().__init__()
 417 |         self.dim = dim
 418 |         self.input_resolution = input_resolution
 419 |         self.depth = depth
 420 |         self.use_checkpoint = use_checkpoint
 421 | 
 422 |         # build blocks
 423 |         self.blocks = nn.ModuleList([
 424 |             TinyViTBlock(dim=dim, input_resolution=input_resolution,
 425 |                          num_heads=num_heads, window_size=window_size,
 426 |                          mlp_ratio=mlp_ratio,
 427 |                          drop=drop,
 428 |                          drop_path=drop_path[i] if isinstance(
 429 |                              drop_path, list) else drop_path,
 430 |                          local_conv_size=local_conv_size,
 431 |                          activation=activation,
 432 |                          )
 433 |             for i in range(depth)])
 434 | 
 435 |         # patch merging layer
 436 |         if downsample is not None:
 437 |             self.downsample = downsample(
 438 |                 input_resolution, dim=dim, out_dim=out_dim, activation=activation)
 439 |         else:
 440 |             self.downsample = None
 441 | 
 442 |     def forward(self, x):
 443 |         for blk in self.blocks:
 444 |             if self.use_checkpoint:
 445 |                 x = checkpoint.checkpoint(blk, x)
 446 |             else:
 447 |                 x = blk(x)
 448 |         if self.downsample is not None:
 449 |             x = self.downsample(x)
 450 |         return x
 451 | 
 452 |     def extra_repr(self) -> str:
 453 |         return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"
 454 | 
 455 | class LayerNorm2d(nn.Module):
 456 |     def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
 457 |         super().__init__()
 458 |         self.weight = nn.Parameter(torch.ones(num_channels))
 459 |         self.bias = nn.Parameter(torch.zeros(num_channels))
 460 |         self.eps = eps
 461 | 
 462 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
 463 |         u = x.mean(1, keepdim=True)
 464 |         s = (x - u).pow(2).mean(1, keepdim=True)
 465 |         x = (x - u) / torch.sqrt(s + self.eps)
 466 |         x = self.weight[:, None, None] * x + self.bias[:, None, None]
 467 |         return x
 468 | 
 469 | class TinyViT(nn.Module):
 470 |     def __init__(self,
 471 |                  img_size=224,
 472 |                  in_chans=3,
 473 |                  #num_classes=1000,
 474 |                  embed_dims=[96, 192, 384, 768], depths=[2, 2, 6, 2],
 475 |                  num_heads=[3, 6, 12, 24],
 476 |                  window_sizes=[7, 7, 14, 7],
 477 |                  mlp_ratio=4.,
 478 |                  drop_rate=0.,
 479 |                  drop_path_rate=0.1,
 480 |                  use_checkpoint=False,
 481 |                  mbconv_expand_ratio=4.0,
 482 |                  local_conv_size=3,
 483 |                  layer_lr_decay=1.0,
 484 |                  ):
 485 |         super().__init__()
 486 |         self.img_size=img_size
 487 |         #self.num_classes = num_classes
 488 |         self.depths = depths
 489 |         self.num_layers = len(depths)
 490 |         self.mlp_ratio = mlp_ratio
 491 | 
 492 |         activation = nn.GELU
 493 | 
 494 |         self.patch_embed = PatchEmbed(in_chans=in_chans,
 495 |                                       embed_dim=embed_dims[0],
 496 |                                       resolution=img_size,
 497 |                                       activation=activation)
 498 | 
 499 |         patches_resolution = self.patch_embed.patches_resolution
 500 |         self.patches_resolution = patches_resolution
 501 | 
 502 |         # stochastic depth
 503 |         dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
 504 |                                                 sum(depths))]  # stochastic depth decay rule
 505 | 
 506 |         # build layers
 507 |         self.layers = nn.ModuleList()
 508 |         for i_layer in range(self.num_layers):
 509 |             kwargs = dict(dim=embed_dims[i_layer],
 510 |                         input_resolution=(
 511 |                             patches_resolution[0] // (2 ** (i_layer-1 if i_layer == 3 else i_layer)),
 512 |                             patches_resolution[1] // (2 ** (i_layer-1 if i_layer == 3 else i_layer))
 513 |                         ),
 514 |                         #   input_resolution=(patches_resolution[0] // (2 ** i_layer),
 515 |                         #                     patches_resolution[1] // (2 ** i_layer)),
 516 |                           depth=depths[i_layer],
 517 |                           drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
 518 |                           downsample=PatchMerging if (
 519 |                               i_layer < self.num_layers - 1) else None,
 520 |                           use_checkpoint=use_checkpoint,
 521 |                           out_dim=embed_dims[min(
 522 |                               i_layer + 1, len(embed_dims) - 1)],
 523 |                           activation=activation,
 524 |                           )
 525 |             if i_layer == 0:
 526 |                 layer = ConvLayer(
 527 |                     conv_expand_ratio=mbconv_expand_ratio,
 528 |                     **kwargs,
 529 |                 )
 530 |             else:
 531 |                 layer = BasicLayer(
 532 |                     num_heads=num_heads[i_layer],
 533 |                     window_size=window_sizes[i_layer],
 534 |                     mlp_ratio=self.mlp_ratio,
 535 |                     drop=drop_rate,
 536 |                     local_conv_size=local_conv_size,
 537 |                     **kwargs)
 538 |             self.layers.append(layer)
 539 | 
 540 |         # init weights
 541 |         self.apply(self._init_weights)
 542 |         self.set_layer_lr_decay(layer_lr_decay)
 543 | 
 544 |         self.neck = nn.Sequential(
 545 |             nn.Conv2d(
 546 |                 embed_dims[-1],
 547 |                 256,
 548 |                 kernel_size=1,
 549 |                 bias=False,
 550 |             ),
 551 |             LayerNorm2d(256),
 552 |             nn.Conv2d(
 553 |                 256,
 554 |                 256,
 555 |                 kernel_size=3,
 556 |                 padding=1,
 557 |                 bias=False,
 558 |             ),
 559 |             LayerNorm2d(256),
 560 |         )
 561 | 
 562 |     def set_layer_lr_decay(self, layer_lr_decay):
 563 |         decay_rate = layer_lr_decay
 564 | 
 565 |         # layers -> blocks (depth)
 566 |         depth = sum(self.depths)
 567 |         lr_scales = [decay_rate ** (depth - i - 1) for i in range(depth)]
 568 | 
 569 |         def _set_lr_scale(m, scale):
 570 |             for p in m.parameters():
 571 |                 p.lr_scale = scale
 572 | 
 573 |         self.patch_embed.apply(lambda x: _set_lr_scale(x, lr_scales[0]))
 574 |         i = 0
 575 |         for layer in self.layers:
 576 |             for block in layer.blocks:
 577 |                 block.apply(lambda x: _set_lr_scale(x, lr_scales[i]))
 578 |                 i += 1
 579 |             if layer.downsample is not None:
 580 |                 layer.downsample.apply(
 581 |                     lambda x: _set_lr_scale(x, lr_scales[i - 1]))
 582 |         assert i == depth
 583 | 
 584 |         for k, p in self.named_parameters():
 585 |             p.param_name = k
 586 | 
 587 |         def _check_lr_scale(m):
 588 |             for p in m.parameters():
 589 |                 assert hasattr(p, 'lr_scale'), p.param_name
 590 | 
 591 |         self.apply(_check_lr_scale)
 592 | 
 593 |     def _init_weights(self, m):
 594 |         if isinstance(m, nn.Linear):
 595 |             trunc_normal_(m.weight, std=.02)
 596 |             if isinstance(m, nn.Linear) and m.bias is not None:
 597 |                 nn.init.constant_(m.bias, 0)
 598 |         elif isinstance(m, nn.LayerNorm):
 599 |             nn.init.constant_(m.bias, 0)
 600 |             nn.init.constant_(m.weight, 1.0)
 601 | 
 602 |     @torch.jit.ignore
 603 |     def no_weight_decay_keywords(self):
 604 |         return {'attention_biases'}
 605 | 
 606 |     def forward_features(self, x):
 607 |         # x: (N, C, H, W)
 608 |         x = self.patch_embed(x)
 609 | 
 610 |         x = self.layers[0](x)
 611 |         start_i = 1
 612 | 
 613 |         for i in range(start_i, len(self.layers)):
 614 |             layer = self.layers[i]
 615 |             x = layer(x)
 616 | 
 617 |         B, _, C = x.size()
 618 |         x = x.view(B, 64, 64, C)
 619 |         x = x.permute(0, 3, 1, 2)
 620 |         x = self.neck(x)
 621 | 
 622 |         return x
 623 | 
 624 |     def forward(self, x):
 625 |         x = self.forward_features(x)
 626 |         return x
```
