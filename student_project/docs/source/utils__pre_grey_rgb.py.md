# utils/pre_grey_rgb.py

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
  14 | save_suffix = '.npy' 
  15 | image_size = 1024
  16 | img_path = 'path to /images' # path to the images
  17 | gt_path = 'path to/labels' # path to the corresponding annotations
  18 | npy_path = 'path to/data/npy/' + prefix[:-1] # save npy path e.g., MedSAM/data/npy/; don't miss the `/`
  19 | os.makedirs(join(npy_path, "gts"), exist_ok=True)
  20 | os.makedirs(join(npy_path, "imgs"), exist_ok=True)
  21 | names = sorted(os.listdir(gt_path))
  22 | print(f'ori \# files {len(names)=}')
  23 | 
  24 | # set label ids that are excluded
  25 | remove_label_ids = [] 
  26 | tumor_id = None # only set this when there are multiple tumors in one image; convert semantic masks to instance masks
  27 | label_id_offset = 0
  28 | do_intensity_cutoff = False # True for grey images
  29 | #%% save preprocessed images and masks as npz files
  30 | for name in tqdm(names):
  31 |     image_name = name.split(gt_name_suffix)[0] + img_name_suffix
  32 |     gt_name = name
  33 |     npy_save_name = prefix + gt_name.split(gt_name_suffix)[0]+save_suffix
  34 |     gt_data_ori = np.uint8(io.imread(join(gt_path, gt_name)))
  35 |     # remove label ids
  36 |     for remove_label_id in remove_label_ids:
  37 |         gt_data_ori[gt_data_ori==remove_label_id] = 0
  38 |     # label tumor masks as instances and remove from gt_data_ori
  39 |     if tumor_id is not None:
  40 |         tumor_bw = np.uint8(gt_data_ori==tumor_id)
  41 |         gt_data_ori[tumor_bw>0] = 0
  42 |         # label tumor masks as instances
  43 |         tumor_inst, tumor_n = cc3d.connected_components(tumor_bw, connectivity=26, return_N=True)
  44 |         # put the tumor instances back to gt_data_ori
  45 |         gt_data_ori[tumor_inst>0] = tumor_inst[tumor_inst>0] + label_id_offset + 1
  46 |     
  47 |     # crop the ground truth with non-zero slices
  48 |     image_data = io.imread(join(img_path, image_name))
  49 |     if np.max(image_data) > 255.0:
  50 |         image_data = np.uint8((image_data-image_data.min()) / (np.max(image_data)-np.min(image_data))*255.0)
  51 |     if len(image_data.shape) == 2:
  52 |         image_data = np.repeat(np.expand_dims(image_data, -1), 3, -1)
  53 |     assert len(image_data.shape) == 3, 'image data is not three channels: img shape:' + str(image_data.shape) + image_name
  54 |     # convert three channel to one channel
  55 |     if image_data.shape[-1] > 3:
  56 |         image_data = image_data[:,:,:3]
  57 |     # image preprocess start
  58 |     if do_intensity_cutoff:
  59 |         lower_bound, upper_bound = np.percentile(image_data[image_data>0], 0.5), np.percentile(image_data[image_data>0], 99.5)
  60 |         image_data_pre = np.clip(image_data, lower_bound, upper_bound)
  61 |         image_data_pre = (image_data_pre - np.min(image_data_pre))/(np.max(image_data_pre)-np.min(image_data_pre))*255.0
  62 |         image_data_pre[image_data==0] = 0
  63 |         image_data_pre = np.uint8(image_data_pre)
  64 |     else:
  65 |         # print('no intensity cutoff')
  66 |         image_data_pre = image_data.copy()
  67 |     np.savez_compressed(join(npy_path, prefix + gt_name.split(gt_name_suffix)[0]+'.npz'), imgs=image_data_pre, gts=gt_data_ori)   
  68 |     resize_img = transform.resize(image_data_pre, (image_size, image_size), order=3, mode='constant', preserve_range=True, anti_aliasing=True)
  69 |     resize_img01 = resize_img/255.0
  70 |     resize_gt = transform.resize(gt_data_ori, (image_size, image_size), order=0, mode='constant', preserve_range=True, anti_aliasing=False)
  71 |     # save resize img and gt as npy
  72 |     np.save(join(npy_path, "imgs", npy_save_name), resize_img01)
  73 |     np.save(join(npy_path, "gts", npy_save_name), resize_gt.astype(np.uint8))
  74 | 
```
