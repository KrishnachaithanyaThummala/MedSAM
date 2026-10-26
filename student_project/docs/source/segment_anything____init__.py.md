# segment_anything/__init__.py

```python
   1 | # -*- coding: utf-8 -*-
   2 | # Copyright (c) Meta Platforms, Inc. and affiliates.
   3 | # All rights reserved.
   4 | 
   5 | # This source code is licensed under the license found in the
   6 | # LICENSE file in the root directory of this source tree.
   7 | 
   8 | from .build_sam import (
   9 |     build_sam,
  10 |     build_sam_vit_h,
  11 |     build_sam_vit_l,
  12 |     build_sam_vit_b,
  13 |     sam_model_registry,
  14 | )
  15 | from .predictor import SamPredictor
  16 | from .automatic_mask_generator import SamAutomaticMaskGenerator
```
