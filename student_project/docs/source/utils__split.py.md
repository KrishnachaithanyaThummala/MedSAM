# utils/split.py

```python
   1 | # -*- coding: utf-8 -*-
   2 | import os
   3 | join = os.path.join
   4 | import random
   5 | 
   6 | path_nii = '' # please complete path; two subfolders: images and labels
   7 | path_video = None # or specify the path
   8 | path_2d = None # or specify the path
   9 | 
  10 | #%% split 3D nii data
  11 | if path_nii is not None:
  12 |     img_path = join(path_nii, 'images')
  13 |     gt_path = join(path_nii, 'labels')
  14 |     gt_names = sorted(os.listdir(gt_path))
  15 |     img_suffix = '_0000.nii.gz'
  16 |     gt_suffix = '.nii.gz'
  17 |     # split 20% data for validation and testing
  18 |     validation_path = join(path_nii, 'validation')
  19 |     os.makedirs(join(validation_path, 'images'), exist_ok=True)
  20 |     os.makedirs(join(validation_path, 'labels'), exist_ok=True)
  21 |     testing_path = join(path_nii, 'testing')
  22 |     os.makedirs(join(testing_path, 'images'), exist_ok=True)
  23 |     os.makedirs(join(testing_path, 'labels'), exist_ok=True)
  24 |     candidates = random.sample(gt_names, int(len(gt_names)*0.2))
  25 |     # split half of test names for validation
  26 |     validation_names = random.sample(candidates, int(len(candidates)*0.5))
  27 |     test_names = [name for name in candidates if name not in validation_names]
  28 |     # move validation and testing data to corresponding folders
  29 |     for name in validation_names:
  30 |         img_name = name.split(gt_suffix)[0] + img_suffix
  31 |         os.rename(join(img_path, img_name), join(validation_path, 'images', img_name))
  32 |         os.rename(join(gt_path, name), join(validation_path, 'labels', name))
  33 |     for name in test_names:
  34 |         img_name = name.split(gt_suffix)[0] + img_suffix
  35 |         os.rename(join(img_path, img_name), join(testing_path, 'images', img_name))
  36 |         os.rename(join(gt_path, name), join(testing_path, 'labels', name))
  37 | 
  38 | 
  39 | ##% split 2D images
  40 | if path_2d is not None:
  41 |     img_path = join(path_2d, 'images')
  42 |     gt_path = join(path_2d, 'labels')
  43 |     gt_names = sorted(os.listdir(gt_path))
  44 |     img_suffix = '.png'
  45 |     gt_suffix = '.png'
  46 |     # split 20% data for validation and testing
  47 |     validation_path = join(path_2d, 'validation')
  48 |     os.makedirs(join(validation_path, 'images'), exist_ok=True)
  49 |     os.makedirs(join(validation_path, 'labels'), exist_ok=True)
  50 |     testing_path = join(path_2d, 'testing')
  51 |     os.makedirs(join(testing_path, 'images'), exist_ok=True)
  52 |     os.makedirs(join(testing_path, 'labels'), exist_ok=True)
  53 |     candidates = random.sample(gt_names, int(len(gt_names)*0.2))
  54 |     # split half of test names for validation
  55 |     validation_names = random.sample(candidates, int(len(candidates)*0.5))
  56 |     test_names = [name for name in candidates if name not in validation_names]
  57 |     # move validation and testing data to corresponding folders
  58 |     for name in validation_names:
  59 |         img_name = name.split(gt_suffix)[0] + img_suffix
  60 |         os.rename(join(img_path, img_name), join(validation_path, 'images', img_name))
  61 |         os.rename(join(gt_path, name), join(validation_path, 'labels', name))
  62 | 
  63 |     for name in test_names:
  64 |         img_name = name.split(gt_suffix)[0] + img_suffix
  65 |         os.rename(join(img_path, img_name), join(testing_path, 'images', img_name))
  66 |         os.rename(join(gt_path, name), join(testing_path, 'labels', name))
  67 | 
  68 | #%% split video data
  69 | if path_video is not None:
  70 |     img_path = join(path_video, 'images')
  71 |     gt_path = join(path_video, 'labels')
  72 |     gt_folders = sorted(os.listdir(gt_path))
  73 |     # split 20% videos for validation and testing
  74 |     validation_path = join(path_video, 'validation')
  75 |     os.makedirs(join(validation_path, 'images'), exist_ok=True)
  76 |     os.makedirs(join(validation_path, 'labels'), exist_ok=True)
  77 |     testing_path = join(path_video, 'testing')
  78 |     os.makedirs(join(testing_path, 'images'), exist_ok=True)
  79 |     os.makedirs(join(testing_path, 'labels'), exist_ok=True)
  80 |     candidates = random.sample(gt_folders, int(len(gt_folders)*0.2))
  81 |     # split half of test names for validation
  82 |     validation_names = random.sample(candidates, int(len(candidates)*0.5))
  83 |     test_names = [name for name in candidates if name not in validation_names]
  84 |     # move validation and testing data to corresponding folders
  85 |     for name in validation_names:
  86 |         os.rename(join(img_path, name), join(validation_path, 'images', name))
  87 |         os.rename(join(gt_path, name), join(validation_path, 'labels', name))
  88 |     for name in test_names:
  89 |         os.rename(join(img_path, name), join(testing_path, 'images', name))
  90 |         os.rename(join(gt_path, name), join(testing_path, 'labels', name))
```
