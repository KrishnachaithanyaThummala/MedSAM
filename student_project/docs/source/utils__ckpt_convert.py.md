# utils/ckpt_convert.py

```python
   1 | # -*- coding: utf-8 -*-
   2 | import torch
   3 | 
   4 | # %% convert medsam model checkpoint to sam checkpoint format for convenient inference
   5 | sam_ckpt_path = ""
   6 | medsam_ckpt_path = ""
   7 | save_path = ""
   8 | multi_gpu_ckpt = True  # set as True if the model is trained with multi-gpu
   9 | 
  10 | sam_ckpt = torch.load(sam_ckpt_path)
  11 | medsam_ckpt = torch.load(medsam_ckpt_path)
  12 | sam_keys = sam_ckpt.keys()
  13 | for key in sam_keys:
  14 |     if not multi_gpu_ckpt:
  15 |         sam_ckpt[key] = medsam_ckpt["model"][key]
  16 |     else:
  17 |         sam_ckpt[key] = medsam_ckpt["model"]["module." + key]
  18 | 
  19 | torch.save(sam_ckpt, save_path)
```
