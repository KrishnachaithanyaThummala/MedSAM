# pre_grey_rgb.py

```python
   1 | #%% import packages
   2 | import numpy as np
   3 | import os
   4 | join = os.path.join 
   5 | from skimage import io, transform
   6 | from tqdm import tqdm
   7 | 
   8 | # convert 2D data to npy files, including images and corresponding masks
   9 | modality = 'dd' # e.g., 'Dermoscopy 
  10 | anatomy = 'dd'  # e.g., 'SkinCancer'
  11 | img_name_suffix = '.png' 
  12 | gt_name_suffix = '.png' 
  13 | prefix = modality + '_' + anatomy + '_'
  14 | img_path = 'path to /images' # path to the images
  15 | gt_path = 'path to/labels' # path to the corresponding annotations
  16 | npz_path = 'path to/data/npy/' + prefix[:-1] # save npy path e.g., MedSAM/data/npy/; don't miss the `/`
  17 | os.makedirs(join(npz_path, "gts"), exist_ok=True)
  18 | os.makedirs(join(npz_path, "imgs"), exist_ok=True)
  19 | names = sorted(os.listdir(gt_path))
  20 | print(f'ori \# files {len(names)=}')
  21 | 
  22 | # set label ids that are excluded
  23 | remove_label_ids = [] 
  24 | tumor_id = None # only set this when there are multiple tumors in one image; convert semantic masks to instance masks
  25 | label_id_offset = 0
  26 | do_intensity_cutoff = False # True for grey images
  27 | 
  28 | #%% save preprocessed images and masks as npz files
  29 | for name in tqdm(names):
  30 |     image_name = name.split(gt_name_suffix)[0] + img_name_suffix
  31 |     gt_name = name
  32 |     npy_save_name = prefix + gt_name.split(gt_name_suffix)[0]+save_suffix
  33 |     gt_data_ori = np.uint8(io.imread(join(gt_path, gt_name)))
  34 |     # remove label ids
  35 |     for remove_label_id in remove_label_ids:
  36 |         gt_data_ori[gt_data_ori==remove_label_id] = 0
  37 |     # label tumor masks as instances and remove from gt_data_ori
  38 |     if tumor_id is not None:
  39 |         tumor_bw = np.uint8(gt_data_ori==tumor_id)
  40 |         gt_data_ori[tumor_bw>0] = 0
  41 |         # label tumor masks as instances
  42 |         tumor_inst, tumor_n = cc3d.connected_components(tumor_bw, connectivity=26, return_N=True)
  43 |         # put the tumor instances back to gt_data_ori
  44 |         gt_data_ori[tumor_inst>0] = tumor_inst[tumor_inst>0] + label_id_offset + 1
  45 |     
  46 |     # crop the ground truth with non-zero slices
  47 |     image_data = io.imread(join(img_path, image_name))
  48 |     if np.max(image_data) > 255.0:
  49 |         image_data = np.uint8((image_data-image_data.min()) / (np.max(image_data)-np.min(image_data))*255.0)
  50 |     if len(image_data.shape) == 2:
  51 |         image_data = np.repeat(np.expand_dims(image_data, -1), 3, -1)
  52 |     assert len(image_data.shape) == 3, 'image data is not three channels: img shape:' + str(image_data.shape) + image_name
  53 |     # convert three channel to one channel
  54 |     if image_data.shape[-1] > 3:
  55 |         image_data = image_data[:,:,:3]
  56 |     # image preprocess start
  57 |     if do_intensity_cutoff:
  58 |         lower_bound, upper_bound = np.percentile(image_data[image_data>0], 0.5), np.percentile(image_data[image_data>0], 99.5)
  59 |         image_data_pre = np.clip(image_data, lower_bound, upper_bound)
  60 |         image_data_pre = (image_data_pre - np.min(image_data_pre))/(np.max(image_data_pre)-np.min(image_data_pre))*255.0
  61 |         image_data_pre[image_data==0] = 0
  62 |         image_data_pre = np.uint8(image_data_pre)
  63 |     else:
  64 |         # print('no intensity cutoff')
  65 |         image_data_pre = image_data.copy()
  66 |     np.savez_compressed(join(npz_path, prefix + gt_name.split(gt_name_suffix)[0]+'.npz'), imgs=image_data_pre, gts=gt_data_ori)   
```
