# utils/format_convert.py

```python
   1 | # -*- coding: utf-8 -*-
   2 | import os
   3 | join = os.path.join
   4 | import random
   5 | import numpy as np
   6 | from skimage import io
   7 | import SimpleITK as sitk
   8 | 
   9 | 
  10 | def dcm2nii(dcm_path, nii_path):
  11 |     """
  12 |     Convert dicom files to nii files
  13 |     """
  14 |     reader = sitk.ImageSeriesReader()
  15 |     dicom_names = reader.GetGDCMSeriesFileNames(dcm_path)
  16 |     reader.SetFileNames(dicom_names)
  17 |     image = reader.Execute()
  18 |     sitk.WriteImage(image, nii_path)
  19 | 
  20 | def mhd2nii(mhd_path, nii_path):
  21 |     """
  22 |     Convert mhd files to nii files
  23 |     """
  24 |     image = sitk.ReadImage(mhd_path)
  25 |     sitk.WriteImage(image, nii_path)
  26 | 
  27 | def nii2nii(nii_path, nii_gz_path):
  28 |     """
  29 |     Convert nii files to nii.gz files, which can reduce the file size
  30 |     """
  31 |     image = sitk.ReadImage(nii_path)
  32 |     sitk.WriteImage(image, nii_gz_path)
  33 | 
  34 | def nrrd2nii(nrrd_path, nii_path):
  35 |     """
  36 |     Convert nrrd files to nii files
  37 |     """
  38 |     image = sitk.ReadImage(nrrd_path)
  39 |     sitk.WriteImage(image, nii_path)
  40 | 
  41 | def jpg2png(jpg_path, png_path):
  42 |     """
  43 |     Convert jpg files to png files
  44 |     """
  45 |     image = io.imread(jpg_path)
  46 |     io.imsave(png_path, image)
  47 | 
  48 | def patchfy(img, mask, outpath, basename):
  49 |     """
  50 |     Patchfy the image and mask into 1024x1024 patches
  51 |     """
  52 |     image_patch_dir = join(outpath, "images")
  53 |     mask_patch_dir = join(outpath, "labels")
  54 |     os.makedirs(image_patch_dir, exist_ok=True)
  55 |     os.makedirs(mask_patch_dir, exist_ok=True)
  56 |     assert img.shape[:2] == mask.shape
  57 |     patch_height = 1024
  58 |     patch_width = 1024
  59 | 
  60 |     img_height, img_width = img.shape[:2]
  61 |     mask_height, mask_width = mask.shape
  62 | 
  63 |     if img_height % patch_height != 0:
  64 |         img = np.pad(img, ((0, patch_height - img_height % patch_height), (0, 0), (0, 0)), mode="constant")
  65 |     if img_width % patch_width != 0:
  66 |         img = np.pad(img, ((0, 0), (0, patch_width - img_width % patch_width), (0, 0)), mode="constant")
  67 |     if mask_height % patch_height != 0:
  68 |         mask = np.pad(mask, ((0, patch_height - mask_height % patch_height), (0, 0)), mode="constant")
  69 |     if mask_width % patch_width != 0:
  70 |         mask = np.pad(mask, ((0, 0), (0, patch_width - mask_width % patch_width)), mode="constant")
  71 | 
  72 |     assert img.shape[:2] == mask.shape
  73 |     assert img.shape[0] % patch_height == 0
  74 |     assert img.shape[1] % patch_width == 0
  75 |     assert mask.shape[0] % patch_height == 0
  76 |     assert mask.shape[1] % patch_width == 0
  77 | 
  78 |     height_steps = (img_height // patch_height) if img_height % patch_height == 0 else (img_height // patch_height + 1)
  79 |     width_steps = (img_width // patch_width) if img_width % patch_width == 0 else (img_width // patch_width + 1)
  80 | 
  81 |     for i in range(height_steps):
  82 |         for j in range(width_steps):
  83 |             img_patch = img[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width, :]
  84 |             mask_patch = mask[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width]
  85 |             assert img_patch.shape[:2] == mask_patch.shape
  86 |             assert img_patch.shape[0] == patch_height
  87 |             assert img_patch.shape[1] == patch_width
  88 |             print(f"img_patch.shape: {img_patch.shape}, mask_patch.shape: {mask_patch.shape}")
  89 |             img_patch_path = join(image_patch_dir, f"{basename}_{i}_{j}.png")
  90 |             mask_patch_path = join(mask_patch_dir, f"{basename}_{i}_{j}.png")
  91 |             io.imsave(img_patch_path, img_patch)    
  92 |             io.imsave(mask_patch_path, mask_patch)
  93 | 
  94 | 
  95 | def rle_decode(mask_rle, img_shape):
  96 |     """
  97 |     #functions to convert encoding to mask and mask to encoding
  98 |     mask_rle: run-length as string formated (start length)
  99 |     shape: (height,width) of array to return
 100 |     Returns numpy array, 1 - mask, 0 - background
 101 |     """
 102 |     seq = mask_rle.split()
 103 |     starts = np.array(list(map(int, seq[0::2])))
 104 |     lengths = np.array(list(map(int, seq[1::2])))
 105 |     assert len(starts) == len(lengths)
 106 |     ends = starts + lengths
 107 |     img = np.zeros((np.product(img_shape),), dtype=np.uint8)
 108 |     for begin, end in zip(starts, ends):
 109 |         img[begin:end] = 255
 110 |     # https://stackoverflow.com/a/46574906/4521646
 111 |     img.shape = img_shape
 112 |     return img.T
```
