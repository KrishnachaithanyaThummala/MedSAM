# pre_CT_MR.py

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
  11 | from tqdm import tqdm
  12 | import cc3d
  13 | 
  14 | import multiprocessing as mp
  15 | from functools import partial
  16 | 
  17 | import argparse
  18 | 
  19 | parser = argparse.ArgumentParser()
  20 | parser.add_argument("-modality", type=str, default="CT", help="CT or MR, [default: CT]")
  21 | parser.add_argument("-anatomy", type=str, default="Abd",
  22 |                     help="Anaotmy name, [default: Abd]")
  23 | parser.add_argument("-img_name_suffix", type=str, default="_0000.nii.gz",
  24 |                     help="Suffix of the image name, [default: _0000.nii.gz]")
  25 | parser.add_argument("-gt_name_suffix", type=str, default=".nii.gz",
  26 |                     help="Suffix of the ground truth name, [default: .nii.gz]")
  27 | parser.add_argument("-img_path", type=str, default="data/FLARE22Train/images",
  28 |                     help="Path to the nii images, [default: data/FLARE22Train/images]")
  29 | parser.add_argument("-gt_path", type=str, default="data/FLARE22Train/labels",
  30 |                     help="Path to the ground truth, [default: data/FLARE22Train/labels]")
  31 | parser.add_argument("-output_path", type=str, default="data/npz",
  32 |                     help="Path to save the npy files, [default: ./data/npz]")
  33 | parser.add_argument("-num_workers", type=int, default=4,
  34 |                     help="Number of workers, [default: 4]")
  35 | # common windows for CT scans https://radiopaedia.org/articles/windowing-ct
  36 | parser.add_argument("-window_level", type=int, default=40,
  37 |                     help="CT window level, [default: 40]")
  38 | parser.add_argument("-window_width", type=int, default=400,
  39 |                     help="CT window width, [default: 400]")
  40 | parser.add_argument("--save_nii", action="store_true",
  41 |                     help="Save the image and ground truth as nii files for sanity check; they can be removed")
  42 | 
  43 | args = parser.parse_args()
  44 | 
  45 | # convert nii image to npz files, including original image and corresponding masks
  46 | modality = args.modality  # CT or MR
  47 | anatomy = args.anatomy  # anantomy + dataset name
  48 | img_name_suffix = args.img_name_suffix  # "_0000.nii.gz"
  49 | gt_name_suffix = args.gt_name_suffix  # ".nii.gz"
  50 | prefix = modality + "_" + anatomy + "_"
  51 | 
  52 | nii_path = args.img_path  # path to the nii images
  53 | gt_path = args.gt_path  # path to the ground truth
  54 | output_path = args.output_path  # path to save the preprocessed files
  55 | npz_tr_path = join(output_path, "MedSAM_train", prefix[:-1])
  56 | os.makedirs(npz_tr_path, exist_ok=True)
  57 | npz_ts_path = join(output_path, "MedSAM_test", prefix[:-1])
  58 | os.makedirs(npz_ts_path, exist_ok=True)
  59 | 
  60 | num_workers = args.num_workers
  61 | 
  62 | voxel_num_thre2d = 10
  63 | voxel_num_thre3d = 100
  64 | 
  65 | names = sorted(os.listdir(gt_path))
  66 | print(f"ori \# files {len(names)=}")
  67 | names = [
  68 |     name
  69 |     for name in names
  70 |     if os.path.exists(join(nii_path, name.split(gt_name_suffix)[0] + img_name_suffix))
  71 | ]
  72 | print(f"after sanity check \# files {len(names)=}")
  73 | 
  74 | # set label ids that are excluded
  75 | remove_label_ids = [
  76 |     12
  77 | ]  # remove deodenum since it is scattered in the image, which is hard to specify with the bounding box
  78 | tumor_id = None  # only set this when there are multiple tumors; convert semantic masks to instance masks
  79 | # set window level and width
  80 | # https://radiopaedia.org/articles/windowing-ct
  81 | WINDOW_LEVEL = args.window_level # only for CT images
  82 | WINDOW_WIDTH = args.window_width # only for CT images
  83 | 
  84 | save_nii = args.save_nii
  85 | # %% save preprocessed images and masks as npz files
  86 | def preprocess(name, npz_path):
  87 |     """
  88 |     Preprocess the image and ground truth, and save them as npz files
  89 | 
  90 |     Parameters
  91 |     ----------
  92 |     name : str
  93 |         name of the ground truth file
  94 |     npz_path : str
  95 |         path to save the npz files
  96 |     """
  97 |     image_name = name.split(gt_name_suffix)[0] + img_name_suffix
  98 |     gt_name = name
  99 |     gt_sitk = sitk.ReadImage(join(gt_path, gt_name))
 100 |     gt_data_ori = np.uint8(sitk.GetArrayFromImage(gt_sitk))
 101 |     # remove label ids
 102 |     for remove_label_id in remove_label_ids:
 103 |         gt_data_ori[gt_data_ori == remove_label_id] = 0
 104 |     # label tumor masks as instances and remove from gt_data_ori
 105 |     if tumor_id is not None:
 106 |         tumor_bw = np.uint8(gt_data_ori == tumor_id)
 107 |         gt_data_ori[tumor_bw > 0] = 0
 108 |         # label tumor masks as instances
 109 |         tumor_inst, tumor_n = cc3d.connected_components(
 110 |             tumor_bw, connectivity=26, return_N=True
 111 |         )
 112 |         # put the tumor instances back to gt_data_ori
 113 |         gt_data_ori[tumor_inst > 0] = (
 114 |             tumor_inst[tumor_inst > 0] + np.max(gt_data_ori)
 115 |         )
 116 | 
 117 |     # exclude the objects with less than 1000 pixels in 3D
 118 |     gt_data_ori = cc3d.dust(
 119 |         gt_data_ori, threshold=voxel_num_thre3d, connectivity=26, in_place=True
 120 |     )
 121 |     # remove small objects with less than 100 pixels in 2D slices
 122 | 
 123 |     for slice_i in range(gt_data_ori.shape[0]):
 124 |         gt_i = gt_data_ori[slice_i, :, :]
 125 |         # remove small objects with less than 100 pixels
 126 |         # reason: fro such small objects, the main challenge is detection rather than segmentation
 127 |         gt_data_ori[slice_i, :, :] = cc3d.dust(
 128 |             gt_i, threshold=voxel_num_thre2d, connectivity=8, in_place=True
 129 |         )
 130 |     # find non-zero slices
 131 |     z_index, _, _ = np.where(gt_data_ori > 0)
 132 |     z_index = np.unique(z_index)
 133 | 
 134 |     if len(z_index) > 0:
 135 |         # crop the ground truth with non-zero slices
 136 |         gt_roi = gt_data_ori[z_index, :, :]
 137 |         # load image and preprocess
 138 |         img_sitk = sitk.ReadImage(join(nii_path, image_name))
 139 |         image_data = sitk.GetArrayFromImage(img_sitk)
 140 |         # nii preprocess start
 141 |         if modality == "CT":
 142 |             lower_bound = WINDOW_LEVEL - WINDOW_WIDTH / 2
 143 |             upper_bound = WINDOW_LEVEL + WINDOW_WIDTH / 2
 144 |             image_data_pre = np.clip(image_data, lower_bound, upper_bound)
 145 |             image_data_pre = (
 146 |                 (image_data_pre - np.min(image_data_pre))
 147 |                 / (np.max(image_data_pre) - np.min(image_data_pre))
 148 |                 * 255.0
 149 |             )
 150 |         else:
 151 |             lower_bound, upper_bound = np.percentile(
 152 |                 image_data[image_data > 0], 0.5
 153 |             ), np.percentile(image_data[image_data > 0], 99.5)
 154 |             image_data_pre = np.clip(image_data, lower_bound, upper_bound)
 155 |             image_data_pre = (
 156 |                 (image_data_pre - np.min(image_data_pre))
 157 |                 / (np.max(image_data_pre) - np.min(image_data_pre))
 158 |                 * 255.0
 159 |             )
 160 |             image_data_pre[image_data == 0] = 0
 161 | 
 162 |         image_data_pre = np.uint8(image_data_pre)
 163 |         img_roi = image_data_pre[z_index, :, :]
 164 |         np.savez_compressed(join(npz_path, prefix + gt_name.split(gt_name_suffix)[0]+'.npz'), imgs=img_roi, gts=gt_roi, spacing=img_sitk.GetSpacing())
 165 | 
 166 |         # save the image and ground truth as nii files for sanity check;
 167 |         # they can be removed
 168 |         if save_nii:
 169 |             img_roi_sitk = sitk.GetImageFromArray(img_roi)
 170 |             img_roi_sitk.SetSpacing(img_sitk.GetSpacing())
 171 |             sitk.WriteImage(
 172 |                 img_roi_sitk,
 173 |                 join(npz_path, prefix + gt_name.split(gt_name_suffix)[0] + "_img.nii.gz"),
 174 |             )
 175 |             gt_roi_sitk = sitk.GetImageFromArray(gt_roi)
 176 |             gt_roi_sitk.SetSpacing(img_sitk.GetSpacing())
 177 |             sitk.WriteImage(
 178 |                 gt_roi_sitk,
 179 |                 join(npz_path, prefix + gt_name.split(gt_name_suffix)[0] + "_gt.nii.gz"),
 180 |             )
 181 | 
 182 | if __name__ == "__main__":
 183 |     tr_names = names[:40]
 184 |     ts_names = names[40:]
 185 | 
 186 |     preprocess_tr = partial(preprocess, npz_path=npz_tr_path)
 187 |     preprocess_ts = partial(preprocess, npz_path=npz_ts_path)
 188 | 
 189 |     with mp.Pool(num_workers) as p:
 190 |         with tqdm(total=len(tr_names)) as pbar:
 191 |             pbar.set_description("Preprocessing training data")
 192 |             for i, _ in tqdm(enumerate(p.imap_unordered(preprocess_tr, tr_names))):
 193 |                 pbar.update()
 194 |         with tqdm(total=len(ts_names)) as pbar:
 195 |             pbar.set_description("Preprocessing testing data")
 196 |             for i, _ in tqdm(enumerate(p.imap_unordered(preprocess_ts, ts_names))):
 197 |                 pbar.update()
```
