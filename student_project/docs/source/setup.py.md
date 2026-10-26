# setup.py

```python
   1 | # Copyright (c) Meta Platforms, Inc. and affiliates.
   2 | # All rights reserved.
   3 | 
   4 | # This source code is licensed under the license found in the
   5 | # LICENSE file in the root directory of this source tree.
   6 | # Adapted from https://github.com/facebookresearch/segment-anything
   7 | 
   8 | from setuptools import find_packages, setup
   9 | 
  10 | setup(
  11 |     name="litemedsam",
  12 |     version="0.0.1",
  13 |     author="Jun Ma",
  14 |     python_requires=">=3.9",
  15 |     install_requires=["monai", "timm", "matplotlib", "scikit-image", "SimpleITK>=2.2.1", "pyarrow", "pandas", "nibabel", "tqdm", "scipy", "ipympl", "opencv-python", "jupyterlab", "ipywidgets"],
  16 |     packages=find_packages(exclude="notebooks"),
  17 |     extras_require={
  18 |         "all": ["pycocotools", "opencv-python", "onnx", "onnxruntime"],
  19 |         "dev": ["flake8", "isort", "black", "mypy"],
  20 |     },
  21 | )
```
