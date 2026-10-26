# utils/extract_weights.py

```python
   1 | # %%
   2 | import torch
   3 | import argparse
   4 | import os
   5 | join = os.path.join
   6 | 
   7 | # %%
   8 | parser = argparse.ArgumentParser()
   9 | parser.add_argument("-from_pth", type=str,
  10 |                     help="Path to the .pth file from which the weights will be extracted")
  11 | parser.add_argument("-to_pth", type=str,
  12 |                     help="Path to the .pth file to which the weights will be saved")
  13 | args = parser.parse_args()
  14 | 
  15 | # %%
  16 | from_pth = args.from_pth
  17 | to_pth = args.to_pth
  18 | 
  19 | # %%
  20 | from_pth = torch.load(from_pth, map_location='cpu')
  21 | assert "model" in from_pth.keys(), "The .pth file does not contain the model weights"
  22 | weights = from_pth["model"]
  23 | torch.save(weights, to_pth)
  24 | print("Weights are saved to {}".format(to_pth))
```
