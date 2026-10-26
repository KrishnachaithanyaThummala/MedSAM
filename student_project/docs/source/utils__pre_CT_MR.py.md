# utils/pre_CT_MR.py

```python
   1 | # -*- coding: utf-8 -*-
   2 | # %% import packages
   3 | # pip install connected-components-3d
   4 | import numpy as np
   5 | 
   6 | # import nibabel as nib
   7 | import SimpleITK as sitk
   8 | import os
   9 | 
  10 | join = os.path.join
  11 | from skimage import transform
  12 | from tqdm import tqdm
  13 | import cc3d
  14 | 
  15 | # convert nii image to npz files, including original image and corresponding masks
  16 | modality = "CT"
  17 | anatomy = "Abd"  # anantomy + dataset name
  18 | img_name_suffix = "_0000.nii.gz"
  19 | gt_name_suffix = ".nii.gz"
  20 | prefix = modality + "_" + anatomy + "_"
  21 | 
  22 | nii_path = "data/FLARE22Train/images"  # path to the nii images
  23 | gt_path = "data/FLARE22Train/labels"  # path to the ground truth
  24 | npy_path = "data/npy/" + prefix[:-1]
  25 | os.makedirs(join(npy_path, "gts"), exist_ok=True)
  26 | os.makedirs(join(npy_path, "imgs"), exist_ok=True)
  27 | 
  28 | image_size = 1024
  29 | voxel_num_thre2d = 100
  30 | voxel_num_thre3d = 1000
  31 | 
  32 | names = sorted(os.listdir(gt_path))
  33 | print(f"ori \# files {len(names)=}")
  34 | names = [
  35 |     name
  36 |     for name in names
  37 |     if os.path.exists(join(nii_path, name.split(gt_name_suffix)[0] + img_name_suffix))
  38 | ]
  39 | print(f"after sanity check \# files {len(names)=}")
  40 | 
  41 | # set label ids that are excluded
  42 | remove_label_ids = [
  43 |     12
  44 | ]  # remove deodenum since it is scattered in the image, which is hard to specify with the bounding box
  45 | tumor_id = None  # only set this when there are multiple tumors; convert semantic masks to instance masks
  46 | # set window level and width
  47 | # https://radiopaedia.org/articles/windowing-ct
  48 | WINDOW_LEVEL = 40  # only for CT images
  49 | WINDOW_WIDTH = 400  # only for CT images
  50 | 
  51 | # %% save preprocessed images and masks as npz files
  52 | for name in tqdm(names[:40]):  # use the remaining 10 cases for validation
  53 |     image_name = name.split(gt_name_suffix)[0] + img_name_suffix
  54 |     gt_name = name
  55 |     gt_sitk = sitk.ReadImage(join(gt_path, gt_name))
  56 |     gt_data_ori = np.uint8(sitk.GetArrayFromImage(gt_sitk))
  57 |     # remove label ids
  58 |     for remove_label_id in remove_label_ids:
  59 |         gt_data_ori[gt_data_ori == remove_label_id] = 0
  60 |     # label tumor masks as instances and remove from gt_data_ori
  61 |     if tumor_id is not None:
  62 |         tumor_bw = np.uint8(gt_data_ori == tumor_id)
  63 |         gt_data_ori[tumor_bw > 0] = 0
  64 |         # label tumor masks as instances
  65 |         tumor_inst, tumor_n = cc3d.connected_components(
  66 |             tumor_bw, connectivity=26, return_N=True
  67 |         )
  68 |         # put the tumor instances back to gt_data_ori
  69 |         gt_data_ori[tumor_inst > 0] = (
  70 |             tumor_inst[tumor_inst > 0] + np.max(gt_data_ori) + 1
  71 |         )
  72 | 
  73 |     # exclude the objects with less than 1000 pixels in 3D
  74 |     gt_data_ori = cc3d.dust(
  75 |         gt_data_ori, threshold=voxel_num_thre3d, connectivity=26, in_place=True
  76 |     )
  77 |     # remove small objects with less than 100 pixels in 2D slices
  78 | 
  79 |     for slice_i in range(gt_data_ori.shape[0]):
  80 |         gt_i = gt_data_ori[slice_i, :, :]
  81 |         # remove small objects with less than 100 pixels
  82 |         # reason: fro such small objects, the main challenge is detection rather than segmentation
  83 |         gt_data_ori[slice_i, :, :] = cc3d.dust(
  84 |             gt_i, threshold=voxel_num_thre2d, connectivity=8, in_place=True
  85 |         )
  86 |     # find non-zero slices
  87 |     z_index, _, _ = np.where(gt_data_ori > 0)
  88 |     z_index = np.unique(z_index)
  89 | 
  90 |     if len(z_index) > 0:
  91 |         # crop the ground truth with non-zero slices
  92 |         gt_roi = gt_data_ori[z_index, :, :]
  93 |         # load image and preprocess
  94 |         img_sitk = sitk.ReadImage(join(nii_path, image_name))
  95 |         image_data = sitk.GetArrayFromImage(img_sitk)
  96 |         # nii preprocess start
  97 |         if modality == "CT":
  98 |             lower_bound = WINDOW_LEVEL - WINDOW_WIDTH / 2
  99 |             upper_bound = WINDOW_LEVEL + WINDOW_WIDTH / 2
 100 |             image_data_pre = np.clip(image_data, lower_bound, upper_bound)
 101 |             image_data_pre = (
 102 |                 (image_data_pre - np.min(image_data_pre))
 103 |                 / (np.max(image_data_pre) - np.min(image_data_pre))
 104 |                 * 255.0
 105 |             )
 106 |         else:
 107 |             lower_bound, upper_bound = np.percentile(
 108 |                 image_data[image_data > 0], 0.5
 109 |             ), np.percentile(image_data[image_data > 0], 99.5)
 110 |             image_data_pre = np.clip(image_data, lower_bound, upper_bound)
 111 |             image_data_pre = (
 112 |                 (image_data_pre - np.min(image_data_pre))
 113 |                 / (np.max(image_data_pre) - np.min(image_data_pre))
 114 |                 * 255.0
 115 |             )
 116 |             image_data_pre[image_data == 0] = 0
 117 | 
 118 |         image_data_pre = np.uint8(image_data_pre)
 119 |         img_roi = image_data_pre[z_index, :, :]
 120 |         np.savez_compressed(join(npy_path, prefix + gt_name.split(gt_name_suffix)[0]+'.npz'), imgs=img_roi, gts=gt_roi, spacing=img_sitk.GetSpacing())
 121 |         # save the image and ground truth as nii files for sanity check;
 122 |         # they can be removed
 123 |         img_roi_sitk = sitk.GetImageFromArray(img_roi)
 124 |         gt_roi_sitk = sitk.GetImageFromArray(gt_roi)
 125 |         sitk.WriteImage(
 126 |             img_roi_sitk,
 127 |             join(npy_path, prefix + gt_name.split(gt_name_suffix)[0] + "_img.nii.gz"),
 128 |         )
 129 |         sitk.WriteImage(
 130 |             gt_roi_sitk,
 131 |             join(npy_path, prefix + gt_name.split(gt_name_suffix)[0] + "_gt.nii.gz"),
 132 |         )
 133 |         # save the each CT image as npy file
 134 |         for i in range(img_roi.shape[0]):
 135 |             img_i = img_roi[i, :, :]
 136 |             img_3c = np.repeat(img_i[:, :, None], 3, axis=-1)
 137 |             resize_img_skimg = transform.resize(
 138 |                 img_3c,
 139 |                 (image_size, image_size),
 140 |                 order=3,
 141 |                 preserve_range=True,
 142 |                 mode="constant",
 143 |                 anti_aliasing=True,
 144 |             )
 145 |             resize_img_skimg_01 = (resize_img_skimg - resize_img_skimg.min()) / np.clip(
 146 |                 resize_img_skimg.max() - resize_img_skimg.min(), a_min=1e-8, a_max=None
 147 |             )  # normalize to [0, 1], (H, W, 3)
 148 |             gt_i = gt_roi[i, :, :]
 149 |             resize_gt_skimg = transform.resize(
 150 |                 gt_i,
 151 |                 (image_size, image_size),
 152 |                 order=0,
 153 |                 preserve_range=True,
 154 |                 mode="constant",
 155 |                 anti_aliasing=False,
 156 |             )
 157 |             resize_gt_skimg = np.uint8(resize_gt_skimg)
 158 |             assert resize_img_skimg_01.shape[:2] == resize_gt_skimg.shape
 159 |             np.save(
 160 |                 join(
 161 |                     npy_path,
 162 |                     "imgs",
 163 |                     prefix
 164 |                     + gt_name.split(gt_name_suffix)[0]
 165 |                     + "-"
 166 |                     + str(i).zfill(3)
 167 |                     + ".npy",
 168 |                 ),
 169 |                 resize_img_skimg_01,
 170 |             )
 171 |             np.save(
 172 |                 join(
 173 |                     npy_path,
 174 |                     "gts",
 175 |                     prefix
 176 |                     + gt_name.split(gt_name_suffix)[0]
 177 |                     + "-"
 178 |                     + str(i).zfill(3)
 179 |                     + ".npy",
 180 |                 ),
 181 |                 resize_gt_skimg,
 182 |             )
```
