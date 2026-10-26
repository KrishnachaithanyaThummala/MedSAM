# LLD_MMRI_step0_data_preprocess.py

```python
   1 | """
   2 | preprocessing code for the LLD-MMRI dataset
   3 | https://github.com/LMMMEng/LLD-MMRI-Dataset
   4 | 
   5 | 1. add spacing and origin to the nii files
   6 | 2. rename the files as caseID-catetory-MRI_phase + '_0000.nii.gz'
   7 | 
   8 | 
   9 | Jun Ma
  10 | """
  11 | 
  12 | # %%
  13 | import glob
  14 | import os
  15 | join = os.path.join
  16 | listdir = os.listdir
  17 | isdir = os.path.isdir
  18 | makedirs = os.makedirs
  19 | dirname = os.path.dirname
  20 | basename = os.path.basename
  21 | from shutil import copyfile
  22 | import SimpleITK as sitk
  23 | from tqdm import tqdm
  24 | import json
  25 | from pprint import pprint
  26 | import pandas as pd
  27 | import re
  28 | from collections import OrderedDict
  29 | from tqdm import tqdm
  30 | import numpy as np
  31 | import multiprocessing as mp
  32 | 
  33 | # %%
  34 | ori_dir = "LLD-MMRI/images"
  35 | json_path = 'LLD-MMRI/labels/Annotation.json'
  36 | save_dir = "LLD-MMRI/images-nii"
  37 | os.makedirs(save_dir, exist_ok=True)
  38 | cases = sorted(listdir(ori_dir))
  39 | 
  40 | with open(json_path) as f:
  41 |     json_data = json.load(f)
  42 | case_names = list(json_data['Annotation_info'].keys())
  43 | # %%
  44 | # case_name = case_names[0]
  45 | def renames(case_name):
  46 |     case_phases = json_data['Annotation_info'][case_name]
  47 |     for case_phase in case_phases:
  48 |         phase = case_phase['phase']
  49 |         studyUID = case_phase['studyUID']
  50 |         seriesUID = case_phase['seriesUID']
  51 |         annotation = case_phase['annotation']
  52 |         pixel_spacing = case_phase['pixel_spacing']
  53 |         slice_spacing = case_phase['slice_spacing']
  54 |         slice_thickness = case_phase['slice_thickness']
  55 |         origin = case_phase['origin']
  56 |         num_targets = annotation['num_targets']
  57 |         catetory = annotation['lesion']['0']['category']
  58 |         
  59 |         sitk_img = sitk.ReadImage(join(ori_dir, case_name, studyUID, seriesUID + '.nii.gz'))
  60 |         # set origin
  61 |         sitk_img.SetOrigin(origin)
  62 |         # set spacing
  63 |         spacing_3d = (pixel_spacing[0], pixel_spacing[1], slice_spacing)
  64 |         sitk_img.SetSpacing(spacing_3d)
  65 |         # set direction
  66 |         direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
  67 |         sitk_img.SetDirection(direction)
  68 | 
  69 |         nii_name = case_name + '_' + str(catetory) + '_' + phase.replace(" ","") + '_0000.nii.gz'
  70 |         sitk.WriteImage(sitk_img, join(save_dir, nii_name))
  71 | 
  72 | 
  73 | if __name__ == "__main__":
  74 |     with mp.Pool(6) as p:
  75 |         r = list(tqdm(p.imap(renames, cases), total=len(cases)))
```
