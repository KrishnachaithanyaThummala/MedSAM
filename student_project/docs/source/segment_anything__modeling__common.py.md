# segment_anything/modeling/common.py

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
  10 | 
  11 | from typing import Type
  12 | 
  13 | 
  14 | class MLPBlock(nn.Module):
  15 |     def __init__(
  16 |         self,
  17 |         embedding_dim: int,
  18 |         mlp_dim: int,
  19 |         act: Type[nn.Module] = nn.GELU,
  20 |     ) -> None:
  21 |         super().__init__()
  22 |         self.lin1 = nn.Linear(embedding_dim, mlp_dim)
  23 |         self.lin2 = nn.Linear(mlp_dim, embedding_dim)
  24 |         self.act = act()
  25 | 
  26 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
  27 |         return self.lin2(self.act(self.lin1(x)))
  28 | 
  29 | 
  30 | # From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
  31 | # Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
  32 | class LayerNorm2d(nn.Module):
  33 |     def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
  34 |         super().__init__()
  35 |         self.weight = nn.Parameter(torch.ones(num_channels))
  36 |         self.bias = nn.Parameter(torch.zeros(num_channels))
  37 |         self.eps = eps
  38 | 
  39 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
  40 |         u = x.mean(1, keepdim=True)
  41 |         s = (x - u).pow(2).mean(1, keepdim=True)
  42 |         x = (x - u) / torch.sqrt(s + self.eps)
  43 |         x = self.weight[:, None, None] * x + self.bias[:, None, None]
  44 |         return x
```
