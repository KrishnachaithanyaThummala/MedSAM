# segment_anything/utils/transforms.py

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
  10 | from torch.nn import functional as F
  11 | from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
  12 | 
  13 | from copy import deepcopy
  14 | from typing import Tuple
  15 | 
  16 | 
  17 | class ResizeLongestSide:
  18 |     """
  19 |     Resizes images to the longest side 'target_length', as well as provides
  20 |     methods for resizing coordinates and boxes. Provides methods for
  21 |     transforming both numpy array and batched torch tensors.
  22 |     """
  23 | 
  24 |     def __init__(self, target_length: int) -> None:
  25 |         self.target_length = target_length
  26 | 
  27 |     def apply_image(self, image: np.ndarray) -> np.ndarray:
  28 |         """
  29 |         Expects a numpy array with shape HxWxC in uint8 format.
  30 |         """
  31 |         target_size = self.get_preprocess_shape(
  32 |             image.shape[0], image.shape[1], self.target_length
  33 |         )
  34 |         return np.array(resize(to_pil_image(image), target_size))
  35 | 
  36 |     def apply_coords(
  37 |         self, coords: np.ndarray, original_size: Tuple[int, ...]
  38 |     ) -> np.ndarray:
  39 |         """
  40 |         Expects a numpy array of length 2 in the final dimension. Requires the
  41 |         original image size in (H, W) format.
  42 |         """
  43 |         old_h, old_w = original_size
  44 |         new_h, new_w = self.get_preprocess_shape(old_h, old_w, self.target_length)
  45 |         new_coords = np.empty_like(coords)
  46 |         new_coords[..., 0] = coords[..., 0] * (new_w / old_w)
  47 |         new_coords[..., 1] = coords[..., 1] * (new_h / old_h)
  48 |         return new_coords
  49 | 
  50 |     def apply_boxes(
  51 |         self, boxes: np.ndarray, original_size: Tuple[int, ...]
  52 |     ) -> np.ndarray:
  53 |         """
  54 |         Expects a numpy array shape Bx4. Requires the original image size
  55 |         in (H, W) format.
  56 |         """
  57 |         boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
  58 |         return boxes.reshape(-1, 4)
  59 | 
  60 |     def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
  61 |         """
  62 |         Expects batched images with shape BxCxHxW and float format. This
  63 |         transformation may not exactly match apply_image. apply_image is
  64 |         the transformation expected by the model.
  65 |         """
  66 |         # Expects an image in BCHW format. May not exactly match apply_image.
  67 |         target_size = self.get_preprocess_shape(
  68 |             image.shape[2], image.shape[3], self.target_length
  69 |         )
  70 |         return F.interpolate(
  71 |             image, target_size, mode="bilinear", align_corners=False, antialias=True
  72 |         )
  73 | 
  74 |     def apply_coords_torch(
  75 |         self, coords: torch.Tensor, original_size: Tuple[int, ...]
  76 |     ) -> torch.Tensor:
  77 |         """
  78 |         Expects a torch tensor with length 2 in the last dimension. Requires the
  79 |         original image size in (H, W) format.
  80 |         """
  81 |         old_h, old_w = original_size
  82 |         new_h, new_w = self.get_preprocess_shape(
  83 |             original_size[0], original_size[1], self.target_length
  84 |         )
  85 |         coords = deepcopy(coords).to(torch.float)
  86 |         coords[..., 0] = coords[..., 0] * (new_w / old_w)
  87 |         coords[..., 1] = coords[..., 1] * (new_h / old_h)
  88 |         return coords
  89 | 
  90 |     def apply_boxes_torch(
  91 |         self, boxes: torch.Tensor, original_size: Tuple[int, ...]
  92 |     ) -> torch.Tensor:
  93 |         """
  94 |         Expects a torch tensor with shape Bx4. Requires the original image
  95 |         size in (H, W) format.
  96 |         """
  97 |         boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
  98 |         return boxes.reshape(-1, 4)
  99 | 
 100 |     @staticmethod
 101 |     def get_preprocess_shape(
 102 |         oldh: int, oldw: int, long_side_length: int
 103 |     ) -> Tuple[int, int]:
 104 |         """
 105 |         Compute the output size given input size and target long side length.
 106 |         """
 107 |         scale = long_side_length * 1.0 / max(oldh, oldw)
 108 |         newh, neww = oldh * scale, oldw * scale
 109 |         neww = int(neww + 0.5)
 110 |         newh = int(newh + 0.5)
 111 |         return (newh, neww)
```
