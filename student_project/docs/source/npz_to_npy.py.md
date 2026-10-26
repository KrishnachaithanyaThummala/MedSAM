# npz_to_npy.py

```python
   1 | """
   2 | Convert the preprocessed .npz files to .npy files for training
   3 | """
   4 | # %% import packages
   5 | import numpy as np
   6 | import os
   7 | join = os.path.join
   8 | listdir = os.listdir
   9 | makedirs = os.makedirs
  10 | from tqdm import tqdm
  11 | import multiprocessing as mp
  12 | import argparse
  13 | 
  14 | parser = argparse.ArgumentParser()
  15 | parser.add_argument("-npz_dir", type=str, default="data/npz/MedSAM_train/CT_Abd",
  16 |                     help="Path to the directory containing preprocessed .npz data, [default: data/npz/MedSAM_train/CT_Abd]")
  17 | parser.add_argument("-npy_dir", type=str, default="data/npy",
  18 |                     help="Path to the directory where the .npy files for training will be saved, [default: ./data/npy]")
  19 | parser.add_argument("-num_workers", type=int, default=4,
  20 |                     help="Number of workers to convert npz to npy in parallel, [default: 4]")
  21 | args = parser.parse_args()
  22 | # %%
  23 | npz_dir = args.npz_dir
  24 | npy_dir = args.npy_dir
  25 | makedirs(join(npy_dir, "imgs"), exist_ok=True)
  26 | makedirs(join(npy_dir, "gts"), exist_ok=True)
  27 | npz_names = [f for f in listdir(npz_dir) if f.endswith(".npz")]
  28 | num_workers = args.num_workers
  29 | # %%
  30 | # convert npz files to npy files
  31 | def convert_npz_to_npy(npz_name):
  32 |     """
  33 |     Convert npz files to npy files for training
  34 | 
  35 |     Parameters
  36 |     ----------
  37 |     npz_name : str
  38 |         Name of the npz file to be converted
  39 |     """
  40 |     name = npz_name.split(".npz")[0]
  41 |     npz_path = join(npz_dir, npz_name)
  42 |     npz = np.load(npz_path, allow_pickle=True, mmap_mode="r")
  43 |     imgs = npz["imgs"]
  44 |     gts = npz["gts"]
  45 |     if len(gts.shape) > 2: ## 3D image
  46 |         for i in range(imgs.shape[0]):
  47 |             img_i = imgs[i, :, :]
  48 |             img_3c = np.repeat(img_i[:, :, None], 3, axis=-1)
  49 | 
  50 |             img_01 = (img_3c - img_3c.min()) / np.clip(
  51 |                 img_3c.max() - img_3c.min(), a_min=1e-8, a_max=None
  52 |             )  # normalize to [0, 1], (H, W, 3)
  53 | 
  54 |             gt_i = gts[i, :, :]
  55 | 
  56 |             gt_i = np.uint8(gt_i)
  57 |             assert img_01.shape[:2] == gt_i.shape
  58 |             np.save(join(npy_dir, "imgs", name + "-" + str(i).zfill(3) + ".npy"), img_01)
  59 |             np.save(join(npy_dir, "gts", name + "-" + str(i).zfill(3) + ".npy"), gt_i)
  60 |     else: ## 2D image
  61 |         if len(imgs.shape) < 3:
  62 |             img_3c = np.repeat(imgs[:, :, None], 3, axis=-1)
  63 |         else:
  64 |             img_3c = imgs
  65 | 
  66 |         img_01 = (img_3c - img_3c.min()) / np.clip(
  67 |             img_3c.max() - img_3c.min(), a_min=1e-8, a_max=None
  68 |         )  # normalize to [0, 1], (H, W, 3)
  69 |         assert img_01.shape[:2] == gts.shape
  70 | 
  71 |         np.save(join(npy_dir, "imgs", name + ".npy"), img_01)
  72 |         np.save(join(npy_dir, "gts", name + ".npy"), gts)
  73 | # %%
  74 | if __name__ == "__main__":
  75 |     with mp.Pool(num_workers) as pool:
  76 |         with tqdm(total=len(npz_names)) as pbar:
  77 |             pbar.set_description("Converting npz to npy")
  78 |             for i, _ in enumerate(pool.imap_unordered(convert_npz_to_npy, npz_names)):
  79 |                 pbar.update()
```
