# evaluation/compute_metrics.py

```python
   1 | # %% move files based on csv file
   2 | import numpy as np
   3 | import nibabel as nb
   4 | import os
   5 | from os import listdir, makedirs
   6 | from os.path import basename, join, dirname, isfile, isdir
   7 | from collections import OrderedDict
   8 | import pandas as pd
   9 | from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
  10 | from tqdm import tqdm
  11 | import multiprocessing as mp
  12 | import argparse
  13 | 
  14 | def compute_multi_class_dsc(gt, seg):
  15 |     dsc = []
  16 |     for i in range(1, gt.max()+1):
  17 |         gt_i = gt == i
  18 |         seg_i = seg == i
  19 |         dsc.append(compute_dice_coefficient(gt_i, seg_i))
  20 |     return np.mean(dsc)
  21 | 
  22 | def compute_multi_class_nsd(gt, seg, spacing, tolerance=2.0):
  23 |     nsd = []
  24 |     for i in range(1, gt.max()+1):
  25 |         gt_i = gt == i
  26 |         seg_i = seg == i
  27 |         surface_distance = compute_surface_distances(
  28 |             gt_i, seg_i, spacing_mm=spacing
  29 |         )
  30 |         nsd.append(compute_surface_dice_at_tolerance(surface_distance, tolerance))
  31 |     return np.mean(nsd)
  32 | 
  33 | 
  34 | parser = argparse.ArgumentParser()
  35 | parser.add_argument('-s', '--seg_dir', default='test_demo/segs', type=str)
  36 | parser.add_argument('-g', '--gt_dir', default='test_demo/gts', type=str)
  37 | parser.add_argument('-csv_dir', default='test_demo/metrics.csv', type=str)
  38 | parser.add_argument('-num_workers', type=int, default=5)
  39 | parser.add_argument('-nsd', default=True, type=bool, help='set it to False to disable NSD computation and save time')
  40 | args = parser.parse_args()
  41 | 
  42 | seg_dir = args.seg_dir
  43 | gt_dir = args.gt_dir
  44 | csv_dir = args.csv_dir
  45 | num_workers = args.num_workers
  46 | compute_NSD = args.nsd
  47 | 
  48 | def compute_metrics(npz_name):
  49 |     metric_dict = {'dsc': -1.}
  50 |     if compute_NSD:
  51 |         metric_dict['nsd'] = -1.
  52 |     npz_seg = np.load(join(seg_dir, npz_name), allow_pickle=True, mmap_mode='r')
  53 |     npz_gt = np.load(join(gt_dir, npz_name), allow_pickle=True, mmap_mode='r')
  54 |     gts = npz_gt['gts']
  55 |     segs = npz_seg['segs']
  56 |     if npz_name.startswith('3D'):
  57 |         spacing = npz_gt['spacing']
  58 |     
  59 |     dsc = compute_multi_class_dsc(gts, segs)
  60 |     # comupute nsd
  61 |     if compute_NSD:
  62 |         if dsc > 0.2:
  63 |         # only compute nsd when dice > 0.2 because NSD is also low when dice is too low
  64 |             if npz_name.startswith('3D'):
  65 |                 nsd = compute_multi_class_nsd(gts, segs, spacing)
  66 |             else:
  67 |                 spacing = [1.0, 1.0, 1.0]
  68 |                 nsd = compute_multi_class_nsd(np.expand_dims(gts, -1), np.expand_dims(segs, -1), spacing)
  69 |         else:
  70 |             nsd = 0.0
  71 |     return npz_name, dsc, nsd
  72 | 
  73 | if __name__ == '__main__':
  74 |     seg_metrics = OrderedDict()
  75 |     seg_metrics['case'] = []
  76 |     seg_metrics['dsc'] = []
  77 |     if compute_NSD:
  78 |         seg_metrics['nsd'] = []
  79 |     
  80 |     npz_names = listdir(gt_dir)
  81 |     npz_names = [npz_name for npz_name in npz_names if npz_name.endswith('.npz')]
  82 |     with mp.Pool(num_workers) as pool:
  83 |         with tqdm(total=len(npz_names)) as pbar:
  84 |             for i, (npz_name, dsc, nsd) in enumerate(pool.imap_unordered(compute_metrics, npz_names)):
  85 |                 seg_metrics['case'].append(npz_name)
  86 |                 seg_metrics['dsc'].append(np.round(dsc, 4))
  87 |                 if compute_NSD:
  88 |                     seg_metrics['nsd'].append(np.round(nsd, 4))
  89 |                 pbar.update()
  90 |     df = pd.DataFrame(seg_metrics)
  91 |     # rank based on case column
  92 |     df = df.sort_values(by=['case'])
  93 |     df.to_csv(csv_dir, index=False)
```
