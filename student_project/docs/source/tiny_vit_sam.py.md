# tiny_vit_sam.py

```python
   1 | # --------------------------------------------------------
   2 | # TinyViT Model Architecture
   3 | # Copyright (c) 2022 Microsoft
   4 | # Adapted from LeViT and Swin Transformer
   5 | #   LeViT: (https://github.com/facebookresearch/levit)
   6 | #   Swin: (https://github.com/microsoft/swin-transformer)
   7 | # Build the TinyViT Model
   8 | # --------------------------------------------------------
   9 | # The TinyViT model is adapted from MobileSAM's variant.
  10 | # --------------------------------------------------------
  11 | 
  12 | import itertools
  13 | import torch
  14 | import torch.nn as nn
  15 | import torch.nn.functional as F
  16 | import torch.utils.checkpoint as checkpoint
  17 | from timm.models.layers import DropPath as TimmDropPath,\
  18 |     to_2tuple, trunc_normal_
  19 | from timm.models.registry import register_model
  20 | from typing import Tuple
  21 | 
  22 | 
  23 | class Conv2d_BN(torch.nn.Sequential):
  24 |     def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
  25 |                  groups=1, bn_weight_init=1):
  26 |         super().__init__()
  27 |         self.add_module('c', torch.nn.Conv2d(
  28 |             a, b, ks, stride, pad, dilation, groups, bias=False))
  29 |         bn = torch.nn.BatchNorm2d(b)
  30 |         torch.nn.init.constant_(bn.weight, bn_weight_init)
  31 |         torch.nn.init.constant_(bn.bias, 0)
  32 |         self.add_module('bn', bn)
  33 | 
  34 |     @torch.no_grad()
  35 |     def fuse(self):
  36 |         c, bn = self._modules.values()
  37 |         w = bn.weight / (bn.running_var + bn.eps)**0.5
  38 |         w = c.weight * w[:, None, None, None]
  39 |         b = bn.bias - bn.running_mean * bn.weight / \
  40 |             (bn.running_var + bn.eps)**0.5
  41 |         m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
  42 |             0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
  43 |         m.weight.data.copy_(w)
  44 |         m.bias.data.copy_(b)
  45 |         return m
  46 | 
  47 | 
  48 | class DropPath(TimmDropPath):
  49 |     def __init__(self, drop_prob=None):
  50 |         super().__init__(drop_prob=drop_prob)
  51 |         self.drop_prob = drop_prob
  52 | 
  53 |     def __repr__(self):
  54 |         msg = super().__repr__()
  55 |         msg += f'(drop_prob={self.drop_prob})'
  56 |         return msg
  57 | 
  58 | 
  59 | class PatchEmbed(nn.Module):
  60 |     def __init__(self, in_chans, embed_dim, resolution, activation):
  61 |         super().__init__()
  62 |         img_size: Tuple[int, int] = to_2tuple(resolution)
  63 |         #self.patches_resolution = (img_size[0] // 4, img_size[1] // 4)
  64 |         self.patches_resolution = img_size
  65 |         self.num_patches = self.patches_resolution[0] * \
  66 |             self.patches_resolution[1]
  67 |         self.in_chans = in_chans
  68 |         self.embed_dim = embed_dim
  69 |         n = embed_dim
  70 |         #self.seq = nn.Sequential(
  71 |         #    Conv2d_BN(in_chans, n // 2, 3, 2, 1),
  72 |         #    activation(),
  73 |         #    Conv2d_BN(n // 2, n, 3, 2, 1),
  74 |         #)
  75 |         self.seq = nn.Sequential(
  76 |             Conv2d_BN(in_chans, n // 2, 1, 1, 0),
  77 |             activation(),
  78 |             Conv2d_BN(n // 2, n, 1, 1, 0),
  79 |         )
  80 | 
  81 |     def forward(self, x):
  82 |         return self.seq(x)
  83 | 
  84 | 
  85 | class MBConv(nn.Module):
  86 |     def __init__(self, in_chans, out_chans, expand_ratio,
  87 |                  activation, drop_path):
  88 |         super().__init__()
  89 |         self.in_chans = in_chans
  90 |         self.hidden_chans = int(in_chans * expand_ratio)
  91 |         self.out_chans = out_chans
  92 | 
  93 |         self.conv1 = Conv2d_BN(in_chans, self.hidden_chans, ks=1)
  94 |         self.act1 = activation()
  95 | 
  96 |         self.conv2 = Conv2d_BN(self.hidden_chans, self.hidden_chans,
  97 |                                ks=3, stride=1, pad=1, groups=self.hidden_chans)
  98 |         self.act2 = activation()
  99 | 
 100 |         self.conv3 = Conv2d_BN(
 101 |             self.hidden_chans, out_chans, ks=1, bn_weight_init=0.0)
 102 |         self.act3 = activation()
 103 | 
 104 |         self.drop_path = DropPath(
 105 |             drop_path) if drop_path > 0. else nn.Identity()
 106 | 
 107 |     def forward(self, x):
 108 |         shortcut = x
 109 | 
 110 |         x = self.conv1(x)
 111 |         x = self.act1(x)
 112 | 
 113 |         x = self.conv2(x)
 114 |         x = self.act2(x)
 115 | 
 116 |         x = self.conv3(x)
 117 | 
 118 |         x = self.drop_path(x)
 119 | 
 120 |         x += shortcut
 121 |         x = self.act3(x)
 122 | 
 123 |         return x
 124 | 
 125 | 
 126 | class PatchMerging(nn.Module):
 127 |     def __init__(self, input_resolution, dim, out_dim, activation):
 128 |         super().__init__()
 129 | 
 130 |         self.input_resolution = input_resolution
 131 |         self.dim = dim
 132 |         self.out_dim = out_dim
 133 |         self.act = activation()
 134 |         self.conv1 = Conv2d_BN(dim, out_dim, 1, 1, 0)
 135 |         stride_c=2
 136 |         if(out_dim==320 or out_dim==448 or out_dim==576):
 137 |             stride_c=1
 138 |         self.conv2 = Conv2d_BN(out_dim, out_dim, 3, stride_c, 1, groups=out_dim)
 139 |         self.conv3 = Conv2d_BN(out_dim, out_dim, 1, 1, 0)
 140 | 
 141 |     def forward(self, x):
 142 |         if x.ndim == 3:
 143 |             H, W = self.input_resolution
 144 |             B = len(x)
 145 |             # (B, C, H, W)
 146 |             x = x.view(B, H, W, -1).permute(0, 3, 1, 2)
 147 | 
 148 |         x = self.conv1(x)
 149 |         x = self.act(x)
 150 | 
 151 |         x = self.conv2(x)
 152 |         x = self.act(x)
 153 |         x = self.conv3(x)
 154 |         x = x.flatten(2).transpose(1, 2)
 155 |         return x
 156 | 
 157 | 
 158 | class ConvLayer(nn.Module):
 159 |     def __init__(self, dim, input_resolution, depth,
 160 |                  activation,
 161 |                  drop_path=0., downsample=None, use_checkpoint=False,
 162 |                  out_dim=None,
 163 |                  conv_expand_ratio=4.,
 164 |                  ):
 165 | 
 166 |         super().__init__()
 167 |         self.dim = dim
 168 |         self.input_resolution = input_resolution
 169 |         self.depth = depth
 170 |         self.use_checkpoint = use_checkpoint
 171 | 
 172 |         # build blocks
 173 |         self.blocks = nn.ModuleList([
 174 |             MBConv(dim, dim, conv_expand_ratio, activation,
 175 |                    drop_path[i] if isinstance(drop_path, list) else drop_path,
 176 |                    )
 177 |             for i in range(depth)])
 178 | 
 179 |         # patch merging layer
 180 |         if downsample is not None:
 181 |             self.downsample = downsample(
 182 |                 input_resolution, dim=dim, out_dim=out_dim, activation=activation)
 183 |         else:
 184 |             self.downsample = None
 185 | 
 186 |     def forward(self, x):
 187 |         for blk in self.blocks:
 188 |             if self.use_checkpoint:
 189 |                 x = checkpoint.checkpoint(blk, x)
 190 |             else:
 191 |                 x = blk(x)
 192 |         if self.downsample is not None:
 193 |             x = self.downsample(x)
 194 |         return x
 195 | 
 196 | 
 197 | class Mlp(nn.Module):
 198 |     def __init__(self, in_features, hidden_features=None,
 199 |                  out_features=None, act_layer=nn.GELU, drop=0.):
 200 |         super().__init__()
 201 |         out_features = out_features or in_features
 202 |         hidden_features = hidden_features or in_features
 203 |         self.norm = nn.LayerNorm(in_features)
 204 |         self.fc1 = nn.Linear(in_features, hidden_features)
 205 |         self.fc2 = nn.Linear(hidden_features, out_features)
 206 |         self.act = act_layer()
 207 |         self.drop = nn.Dropout(drop)
 208 | 
 209 |     def forward(self, x):
 210 |         x = self.norm(x)
 211 | 
 212 |         x = self.fc1(x)
 213 |         x = self.act(x)
 214 |         x = self.drop(x)
 215 |         x = self.fc2(x)
 216 |         x = self.drop(x)
 217 |         return x
 218 | 
 219 | 
 220 | class Attention(torch.nn.Module):
 221 |     def __init__(self, dim, key_dim, num_heads=8,
 222 |                  attn_ratio=4,
 223 |                  resolution=(14, 14),
 224 |                  ):
 225 |         super().__init__()
 226 |         # (h, w)
 227 |         assert isinstance(resolution, tuple) and len(resolution) == 2
 228 |         self.num_heads = num_heads
 229 |         self.scale = key_dim ** -0.5
 230 |         self.key_dim = key_dim
 231 |         self.nh_kd = nh_kd = key_dim * num_heads
 232 |         self.d = int(attn_ratio * key_dim)
 233 |         self.dh = int(attn_ratio * key_dim) * num_heads
 234 |         self.attn_ratio = attn_ratio
 235 |         h = self.dh + nh_kd * 2
 236 | 
 237 |         self.norm = nn.LayerNorm(dim)
 238 |         self.qkv = nn.Linear(dim, h)
 239 |         self.proj = nn.Linear(self.dh, dim)
 240 | 
 241 |         points = list(itertools.product(
 242 |             range(resolution[0]), range(resolution[1])))
 243 |         N = len(points)
 244 |         attention_offsets = {}
 245 |         idxs = []
 246 |         for p1 in points:
 247 |             for p2 in points:
 248 |                 offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
 249 |                 if offset not in attention_offsets:
 250 |                     attention_offsets[offset] = len(attention_offsets)
 251 |                 idxs.append(attention_offsets[offset])
 252 |         self.attention_biases = torch.nn.Parameter(
 253 |             torch.zeros(num_heads, len(attention_offsets)))
 254 |         self.register_buffer('attention_bias_idxs',
 255 |                              torch.LongTensor(idxs).view(N, N),
 256 |                              persistent=False)
 257 | 
 258 |     @torch.no_grad()
 259 |     def train(self, mode=True):
 260 |         super().train(mode)
 261 |         if mode and hasattr(self, 'ab'):
 262 |             del self.ab
 263 |         else:
 264 |             self.register_buffer('ab',
 265 |                                  self.attention_biases[:, self.attention_bias_idxs],
 266 |                                  persistent=False)
 267 | 
 268 |     def forward(self, x):  # x (B,N,C)
 269 |         B, N, _ = x.shape
 270 | 
 271 |         # Normalization
 272 |         x = self.norm(x)
 273 | 
 274 |         qkv = self.qkv(x)
 275 |         # (B, N, num_heads, d)
 276 |         q, k, v = qkv.view(B, N, self.num_heads, -
 277 |                            1).split([self.key_dim, self.key_dim, self.d], dim=3)
 278 |         # (B, num_heads, N, d)
 279 |         q = q.permute(0, 2, 1, 3)
 280 |         k = k.permute(0, 2, 1, 3)
 281 |         v = v.permute(0, 2, 1, 3)
 282 | 
 283 |         attn = (
 284 |             (q @ k.transpose(-2, -1)) * self.scale
 285 |             +
 286 |             (self.attention_biases[:, self.attention_bias_idxs]
 287 |              if self.training else self.ab)
 288 |         )
 289 |         attn = attn.softmax(dim=-1)
 290 |         x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
 291 |         x = self.proj(x)
 292 |         return x
 293 | 
 294 | 
 295 | class TinyViTBlock(nn.Module):
 296 |     r""" TinyViT Block.
 297 | 
 298 |     Args:
 299 |         dim (int): Number of input channels.
 300 |         input_resolution (tuple[int, int]): Input resolution.
 301 |         num_heads (int): Number of attention heads.
 302 |         window_size (int): Window size.
 303 |         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
 304 |         drop (float, optional): Dropout rate. Default: 0.0
 305 |         drop_path (float, optional): Stochastic depth rate. Default: 0.0
 306 |         local_conv_size (int): the kernel size of the convolution between
 307 |                                Attention and MLP. Default: 3
 308 |         activation: the activation function. Default: nn.GELU
 309 |     """
 310 | 
 311 |     def __init__(self, dim, input_resolution, num_heads, window_size=7,
 312 |                  mlp_ratio=4., drop=0., drop_path=0.,
 313 |                  local_conv_size=3,
 314 |                  activation=nn.GELU,
 315 |                  ):
 316 |         super().__init__()
 317 |         self.dim = dim
 318 |         self.input_resolution = input_resolution
 319 |         self.num_heads = num_heads
 320 |         assert window_size > 0, 'window_size must be greater than 0'
 321 |         self.window_size = window_size
 322 |         self.mlp_ratio = mlp_ratio
 323 | 
 324 |         self.drop_path = DropPath(
 325 |             drop_path) if drop_path > 0. else nn.Identity()
 326 | 
 327 |         assert dim % num_heads == 0, 'dim must be divisible by num_heads'
 328 |         head_dim = dim // num_heads
 329 | 
 330 |         window_resolution = (window_size, window_size)
 331 |         self.attn = Attention(dim, head_dim, num_heads,
 332 |                               attn_ratio=1, resolution=window_resolution)
 333 | 
 334 |         mlp_hidden_dim = int(dim * mlp_ratio)
 335 |         mlp_activation = activation
 336 |         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
 337 |                        act_layer=mlp_activation, drop=drop)
 338 | 
 339 |         pad = local_conv_size // 2
 340 |         self.local_conv = Conv2d_BN(
 341 |             dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim)
 342 | 
 343 |     def forward(self, x):
 344 |         H, W = self.input_resolution
 345 |         B, L, C = x.shape
 346 |         assert L == H * W, "input feature has wrong size"
 347 |         res_x = x
 348 |         if H == self.window_size and W == self.window_size:
 349 |             x = self.attn(x)
 350 |         else:
 351 |             x = x.view(B, H, W, C)
 352 |             pad_b = (self.window_size - H %
 353 |                      self.window_size) % self.window_size
 354 |             pad_r = (self.window_size - W %
 355 |                      self.window_size) % self.window_size
 356 |             padding = pad_b > 0 or pad_r > 0
 357 | 
 358 |             if padding:
 359 |                 x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
 360 | 
 361 |             pH, pW = H + pad_b, W + pad_r
 362 |             nH = pH // self.window_size
 363 |             nW = pW // self.window_size
 364 |             # window partition
 365 |             x = x.view(B, nH, self.window_size, nW, self.window_size, C).transpose(2, 3).reshape(
 366 |                 B * nH * nW, self.window_size * self.window_size, C)
 367 |             x = self.attn(x)
 368 |             # window reverse
 369 |             x = x.view(B, nH, nW, self.window_size, self.window_size,
 370 |                        C).transpose(2, 3).reshape(B, pH, pW, C)
 371 | 
 372 |             if padding:
 373 |                 x = x[:, :H, :W].contiguous()
 374 | 
 375 |             x = x.view(B, L, C)
 376 | 
 377 |         x = res_x + self.drop_path(x)
 378 | 
 379 |         x = x.transpose(1, 2).reshape(B, C, H, W)
 380 |         x = self.local_conv(x)
 381 |         x = x.view(B, C, L).transpose(1, 2)
 382 | 
 383 |         x = x + self.drop_path(self.mlp(x))
 384 |         return x
 385 | 
 386 |     def extra_repr(self) -> str:
 387 |         return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
 388 |                f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"
 389 | 
 390 | 
 391 | class BasicLayer(nn.Module):
 392 |     """ A basic TinyViT layer for one stage.
 393 | 
 394 |     Args:
 395 |         dim (int): Number of input channels.
 396 |         input_resolution (tuple[int]): Input resolution.
 397 |         depth (int): Number of blocks.
 398 |         num_heads (int): Number of attention heads.
 399 |         window_size (int): Local window size.
 400 |         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
 401 |         drop (float, optional): Dropout rate. Default: 0.0
 402 |         drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
 403 |         downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
 404 |         use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
 405 |         local_conv_size: the kernel size of the depthwise convolution between attention and MLP. Default: 3
 406 |         activation: the activation function. Default: nn.GELU
 407 |         out_dim: the output dimension of the layer. Default: dim
 408 |     """
 409 | 
 410 |     def __init__(self, dim, input_resolution, depth, num_heads, window_size,
 411 |                  mlp_ratio=4., drop=0.,
 412 |                  drop_path=0., downsample=None, use_checkpoint=False,
 413 |                  local_conv_size=3,
 414 |                  activation=nn.GELU,
 415 |                  out_dim=None,
 416 |                  ):
 417 | 
 418 |         super().__init__()
 419 |         self.dim = dim
 420 |         self.input_resolution = input_resolution
 421 |         self.depth = depth
 422 |         self.use_checkpoint = use_checkpoint
 423 | 
 424 |         # build blocks
 425 |         self.blocks = nn.ModuleList([
 426 |             TinyViTBlock(dim=dim, input_resolution=input_resolution,
 427 |                          num_heads=num_heads, window_size=window_size,
 428 |                          mlp_ratio=mlp_ratio,
 429 |                          drop=drop,
 430 |                          drop_path=drop_path[i] if isinstance(
 431 |                              drop_path, list) else drop_path,
 432 |                          local_conv_size=local_conv_size,
 433 |                          activation=activation,
 434 |                          )
 435 |             for i in range(depth)])
 436 | 
 437 |         # patch merging layer
 438 |         if downsample is not None:
 439 |             self.downsample = downsample(
 440 |                 input_resolution, dim=dim, out_dim=out_dim, activation=activation)
 441 |         else:
 442 |             self.downsample = None
 443 | 
 444 |     def forward(self, x):
 445 |         for blk in self.blocks:
 446 |             if self.use_checkpoint:
 447 |                 x = checkpoint.checkpoint(blk, x)
 448 |             else:
 449 |                 x = blk(x)
 450 |         if self.downsample is not None:
 451 |             x = self.downsample(x)
 452 |         return x
 453 | 
 454 |     def extra_repr(self) -> str:
 455 |         return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"
 456 | 
 457 | class LayerNorm2d(nn.Module):
 458 |     def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
 459 |         super().__init__()
 460 |         self.weight = nn.Parameter(torch.ones(num_channels))
 461 |         self.bias = nn.Parameter(torch.zeros(num_channels))
 462 |         self.eps = eps
 463 | 
 464 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
 465 |         u = x.mean(1, keepdim=True)
 466 |         s = (x - u).pow(2).mean(1, keepdim=True)
 467 |         x = (x - u) / torch.sqrt(s + self.eps)
 468 |         x = self.weight[:, None, None] * x + self.bias[:, None, None]
 469 |         return x
 470 | 
 471 | class TinyViT(nn.Module):
 472 |     def __init__(self,
 473 |                  img_size=224,
 474 |                  in_chans=3,
 475 |                  #num_classes=1000,
 476 |                  embed_dims=[96, 192, 384, 768], depths=[2, 2, 6, 2],
 477 |                  num_heads=[3, 6, 12, 24],
 478 |                  window_sizes=[7, 7, 14, 7],
 479 |                  mlp_ratio=4.,
 480 |                  drop_rate=0.,
 481 |                  drop_path_rate=0.1,
 482 |                  use_checkpoint=False,
 483 |                  mbconv_expand_ratio=4.0,
 484 |                  local_conv_size=3,
 485 |                  layer_lr_decay=1.0,
 486 |                  ):
 487 |         super().__init__()
 488 |         self.img_size=img_size
 489 |         #self.num_classes = num_classes
 490 |         self.depths = depths
 491 |         self.num_layers = len(depths)
 492 |         self.mlp_ratio = mlp_ratio
 493 | 
 494 |         activation = nn.GELU
 495 | 
 496 |         self.patch_embed = PatchEmbed(in_chans=in_chans,
 497 |                                       embed_dim=embed_dims[0],
 498 |                                       resolution=img_size,
 499 |                                       activation=activation)
 500 | 
 501 |         patches_resolution = self.patch_embed.patches_resolution
 502 |         self.patches_resolution = patches_resolution
 503 | 
 504 |         # stochastic depth
 505 |         dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
 506 |                                                 sum(depths))]  # stochastic depth decay rule
 507 | 
 508 |         # build layers
 509 |         self.layers = nn.ModuleList()
 510 |         for i_layer in range(self.num_layers):
 511 |             kwargs = dict(dim=embed_dims[i_layer],
 512 |                         input_resolution=(
 513 |                             patches_resolution[0] // (2 ** (i_layer-1 if i_layer == 3 else i_layer)),
 514 |                             patches_resolution[1] // (2 ** (i_layer-1 if i_layer == 3 else i_layer))
 515 |                         ),
 516 |                         #   input_resolution=(patches_resolution[0] // (2 ** i_layer),
 517 |                         #                     patches_resolution[1] // (2 ** i_layer)),
 518 |                           depth=depths[i_layer],
 519 |                           drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
 520 |                           downsample=PatchMerging if (
 521 |                               i_layer < self.num_layers - 1) else None,
 522 |                           use_checkpoint=use_checkpoint,
 523 |                           out_dim=embed_dims[min(
 524 |                               i_layer + 1, len(embed_dims) - 1)],
 525 |                           activation=activation,
 526 |                           )
 527 |             if i_layer == 0:
 528 |                 layer = ConvLayer(
 529 |                     conv_expand_ratio=mbconv_expand_ratio,
 530 |                     **kwargs,
 531 |                 )
 532 |             else:
 533 |                 layer = BasicLayer(
 534 |                     num_heads=num_heads[i_layer],
 535 |                     window_size=window_sizes[i_layer],
 536 |                     mlp_ratio=self.mlp_ratio,
 537 |                     drop=drop_rate,
 538 |                     local_conv_size=local_conv_size,
 539 |                     **kwargs)
 540 |             self.layers.append(layer)
 541 | 
 542 |         # init weights
 543 |         self.apply(self._init_weights)
 544 |         self.set_layer_lr_decay(layer_lr_decay)
 545 | 
 546 |         self.neck = nn.Sequential(
 547 |             nn.Conv2d(
 548 |                 embed_dims[-1],
 549 |                 256,
 550 |                 kernel_size=1,
 551 |                 bias=False,
 552 |             ),
 553 |             LayerNorm2d(256),
 554 |             nn.Conv2d(
 555 |                 256,
 556 |                 256,
 557 |                 kernel_size=3,
 558 |                 padding=1,
 559 |                 bias=False,
 560 |             ),
 561 |             LayerNorm2d(256),
 562 |         )
 563 | 
 564 |     def set_layer_lr_decay(self, layer_lr_decay):
 565 |         decay_rate = layer_lr_decay
 566 | 
 567 |         # layers -> blocks (depth)
 568 |         depth = sum(self.depths)
 569 |         lr_scales = [decay_rate ** (depth - i - 1) for i in range(depth)]
 570 | 
 571 |         def _set_lr_scale(m, scale):
 572 |             for p in m.parameters():
 573 |                 p.lr_scale = scale
 574 | 
 575 |         self.patch_embed.apply(lambda x: _set_lr_scale(x, lr_scales[0]))
 576 |         i = 0
 577 |         for layer in self.layers:
 578 |             for block in layer.blocks:
 579 |                 block.apply(lambda x: _set_lr_scale(x, lr_scales[i]))
 580 |                 i += 1
 581 |             if layer.downsample is not None:
 582 |                 layer.downsample.apply(
 583 |                     lambda x: _set_lr_scale(x, lr_scales[i - 1]))
 584 |         assert i == depth
 585 | 
 586 |         for k, p in self.named_parameters():
 587 |             p.param_name = k
 588 | 
 589 |         def _check_lr_scale(m):
 590 |             for p in m.parameters():
 591 |                 assert hasattr(p, 'lr_scale'), p.param_name
 592 | 
 593 |         self.apply(_check_lr_scale)
 594 | 
 595 |     def _init_weights(self, m):
 596 |         if isinstance(m, nn.Linear):
 597 |             trunc_normal_(m.weight, std=.02)
 598 |             if isinstance(m, nn.Linear) and m.bias is not None:
 599 |                 nn.init.constant_(m.bias, 0)
 600 |         elif isinstance(m, nn.LayerNorm):
 601 |             nn.init.constant_(m.bias, 0)
 602 |             nn.init.constant_(m.weight, 1.0)
 603 | 
 604 |     @torch.jit.ignore
 605 |     def no_weight_decay_keywords(self):
 606 |         return {'attention_biases'}
 607 | 
 608 |     def forward_features(self, x):
 609 |         # x: (N, C, H, W)
 610 |         x = self.patch_embed(x)
 611 | 
 612 |         x = self.layers[0](x)
 613 |         start_i = 1
 614 | 
 615 |         for i in range(start_i, len(self.layers)):
 616 |             layer = self.layers[i]
 617 |             x = layer(x)
 618 | 
 619 |         B, _, C = x.size()
 620 |         x = x.view(B, 64, 64, C)
 621 |         x = x.permute(0, 3, 1, 2)
 622 |         x = self.neck(x)
 623 | 
 624 |         return x
 625 | 
 626 |     def forward(self, x):
 627 |         x = self.forward_features(x)
 628 |         return x
```
