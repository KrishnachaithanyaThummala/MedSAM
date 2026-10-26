# distillation/distill_single_gpu.py

```python
   1 | # %%
   2 | import os
   3 | import glob
   4 | import random
   5 | import argparse
   6 | from os import listdir, makedirs
   7 | from os.path import join, exists, isfile, isdir
   8 | from tqdm import tqdm
   9 | from copy import deepcopy
  10 | from time import time
  11 | import numpy as np
  12 | import torch
  13 | import torch.nn as nn
  14 | import torch.optim as optim
  15 | from torch.utils.data import Dataset, DataLoader
  16 | from segment_anything import sam_model_registry
  17 | from matplotlib import pyplot as plt
  18 | import cv2
  19 | import torch.nn.functional as F
  20 | 
  21 | from tiny_vit_sam import TinyViT
  22 | 
  23 | # %%
  24 | parser = argparse.ArgumentParser()
  25 | parser.add_argument('-i', '--tr_npy_path', type=str,
  26 |                     default=None,
  27 |                     help='path to training npy files; two subfolders: gts and imgs')
  28 | parser.add_argument('-task_name', type=str, default='MedSAM-Lite')
  29 | parser.add_argument('-medsam_checkpoint', type=str,
  30 |                     default=None,
  31 |                     help='path to MedSAM encoder checkpoint')
  32 | parser.add_argument('-tinyvit_checkpoint', type=str,
  33 |                     default=None,
  34 |                     help='path to TinyViT encoder checkpoint (not required)')
  35 | parser.add_argument('-work_dir', type=str, default='./work_dir',
  36 |                     help='path to save the model checkpoints and logs')
  37 | parser.add_argument('--data_aug', type=bool, default=True, 
  38 |                     help='use data augmentation during training')
  39 | # train
  40 | parser.add_argument('-num_epochs', type=int, default=1000)
  41 | parser.add_argument('-batch_size', type=int, default=8)
  42 | parser.add_argument('-num_workers', type=int, default=8)
  43 | # Optimizer parameters
  44 | parser.add_argument('-weight_decay', type=float, default=0.01,
  45 |                     help='weight decay (default: 0.01)')
  46 | parser.add_argument('-lr', type=float, default=5e-4, metavar='LR',
  47 |                     help='learning rate (absolute lr)')
  48 | parser.add_argument('-resume', type = str,
  49 |                     default = None,
  50 |                     help="Resuming training from saved checkpoint (only required when resuming training)")
  51 | 
  52 | args = parser.parse_args()
  53 | 
  54 | # %%
  55 | data_root = args.tr_npy_path
  56 | checkpoint = args.medsam_checkpoint
  57 | medsam_tiny_encoder_ckpt_init = args.tinyvit_checkpoint
  58 | ckpt_dir = args.work_dir
  59 | log_dir = args.work_dir
  60 | max_epochs = args.num_epochs
  61 | batch_size = args.batch_size
  62 | num_workers = args.batch_size
  63 | device = "cuda:0"
  64 | lr = args.lr
  65 | makedirs(ckpt_dir, exist_ok=True)
  66 | makedirs(log_dir, exist_ok=True)
  67 | 
  68 | # %%
  69 | class Logger(object):
  70 |     def __init__(self):
  71 |         self.logging = {
  72 |             'train_losses': [],
  73 |             'lrs': [],
  74 |             'epoch_start_timestamps': [],
  75 |             'epoch_end_timestamps': []
  76 |         }
  77 | 
  78 |     def log(self, key, value):
  79 |         self.logging[key].append(value)
  80 |     
  81 |     def plot_progress_png(self, output_folder):
  82 |         fig, ax_all = plt.subplots(2, 1, figsize=(10, 8))
  83 |         x_values = [i for i in range(1, len(self.logging['train_losses']) + 1)]
  84 |         ax_all[0].plot(
  85 |             x_values,
  86 |             self.logging['train_losses'],
  87 |             #color='b',
  88 |             ls='-',
  89 |             label="loss_tr",
  90 |         )
  91 |         ax_all[0].set_ylabel("loss", fontsize=10)
  92 |         ax_all[0].legend(loc=(0, 1))
  93 | 
  94 |         ax_all[1].plot(x_values, [i - j for i, j in zip(self.logging['epoch_end_timestamps'],
  95 |                                                  self.logging['epoch_start_timestamps'])],
  96 |                                                  #color='b',
  97 |                 ls='-',
  98 |                 label="epoch duration",
  99 |                 )
 100 |         ax_all[1].set_xlabel("epoch", fontsize=10)
 101 |         ax_all[1].set_ylabel("time [s]", fontsize=10)
 102 |         ax_all[1].legend(loc=(0, 1))
 103 | 
 104 |         plt.tight_layout()
 105 | 
 106 |         fig.savefig(join(output_folder, "progress.png"))
 107 |         plt.close()
 108 | 
 109 |     def get_checkpoint(self):
 110 |         return self.logging
 111 | 
 112 |     def load_checkpoint(self, checkpoint: dict):
 113 |         self.logging = checkpoint
 114 | 
 115 | # %%
 116 | class NpyDataset(Dataset): 
 117 |     def __init__(self, data_root, data_aug=True):
 118 |         self.data_root = data_root
 119 |         self.gt_path = join(data_root, 'gts')
 120 |         self.img_path = join(data_root, 'imgs')
 121 |         self.gt_path_files = sorted(glob.glob(join(self.gt_path, '**/*.npy'), recursive=True))
 122 |         self.gt_path_files = [file for file in self.gt_path_files if os.path.isfile(join(self.img_path, os.path.basename(file)))]
 123 |         self.data_aug = data_aug
 124 |         print(f'number of images: {len(self.gt_path_files)}')
 125 |     
 126 |     def __len__(self):
 127 |         return len(self.gt_path_files)
 128 |     
 129 |     def __getitem__(self, index):
 130 |         img_name = os.path.basename(self.gt_path_files[index])
 131 |         img_3c = np.load(join(self.img_path, img_name), 'r', allow_pickle=True) # (H, W, 3)
 132 |         resize_img_skimg = cv2.resize(
 133 |             img_3c, (1024, 1024), interpolation=cv2.INTER_CUBIC
 134 |         )
 135 |         resize_img_skimg_01 = (resize_img_skimg - resize_img_skimg.min()) / np.clip(resize_img_skimg.max() - resize_img_skimg.min(), a_min=1e-8, a_max=None) # normalize to [0, 1], (H, W, 3)
 136 |         # convert the shape to (3, H, W)
 137 |         img_1024 = np.transpose(resize_img_skimg_01, (2, 0, 1))
 138 |         assert np.max(img_1024)<=1.0 and np.min(img_1024)>=0.0, 'image should be normalized to [0, 1]'
 139 | 
 140 |         img_256 = cv2.resize(
 141 |             img_3c, (256, 256), interpolation=cv2.INTER_AREA
 142 |         )
 143 |         img_256 = (img_256 - img_256.min()) / np.clip(
 144 |             img_256.max() - img_256.min(), a_min=1e-8, a_max=None
 145 |         )
 146 |         img_256 = np.transpose(img_256, (2, 0, 1))
 147 |         assert np.max(img_256)<=1.0 and np.min(img_256)>=0.0, 'image should be normalized to [0, 1]'
 148 | 
 149 |         if self.data_aug:
 150 |             if random.random() > 0.5:
 151 |                 img_1024 = np.ascontiguousarray(np.flip(img_1024, axis=-1))
 152 |                 img_256 = np.ascontiguousarray(np.flip(img_256, axis=-1))
 153 |                 # print('DA with flip left right')
 154 |             if random.random() > 0.5:
 155 |                 img_1024 = np.ascontiguousarray(np.flip(img_1024, axis=-2))
 156 |                 img_256 = np.ascontiguousarray(np.flip(img_256, axis=-2))
 157 |                 # print('DA with flip up down')
 158 | 
 159 |         return torch.tensor(img_1024).float(), torch.tensor(img_256).float(), img_name
 160 | 
 161 | medsam_model = sam_model_registry["vit_b"](checkpoint=checkpoint)
 162 | teacher_model = deepcopy(medsam_model.image_encoder)
 163 | teacher_model = teacher_model.to(device)
 164 | teacher_model.eval()
 165 | del medsam_model
 166 | torch.cuda.empty_cache()
 167 | for param in teacher_model.parameters():
 168 |     param.requires_grad = False ## freeze teacher model
 169 | # %%
 170 | student_model = TinyViT(
 171 |     img_size=256,
 172 |     in_chans=3,
 173 |     embed_dims=[
 174 |         64,
 175 |         128,
 176 |         160,
 177 |         320 
 178 |     ],
 179 |     depths=[2, 2, 6, 2],
 180 |     num_heads=[2, 4, 5, 10],
 181 |     window_sizes=[7, 7, 14, 7],
 182 |     mlp_ratio=4.,
 183 |     drop_rate=0.,
 184 |     drop_path_rate=0.0,
 185 |     use_checkpoint=False,
 186 |     mbconv_expand_ratio=4.0,
 187 |     local_conv_size=3,
 188 |     layer_lr_decay=0.8
 189 | ) ## TinyViT setup
 190 | 
 191 | # %%
 192 | pretrained_student_ckpt = torch.load(
 193 |     medsam_tiny_encoder_ckpt_init,
 194 |     map_location="cpu"
 195 | )
 196 | student_model.load_state_dict(pretrained_student_ckpt)
 197 | student_model = student_model.to(device)
 198 | # %%
 199 | print(f"MedSAM encoder size: {sum(p.numel() for p in teacher_model.parameters())}")
 200 | print(f"TinyViT encoder size: {sum(p.numel() for p in student_model.parameters())}")
 201 | 
 202 | logger = Logger()
 203 | optimizer = optim.AdamW(
 204 |     student_model.parameters(),
 205 |     lr=lr,
 206 |     betas=(0.9, 0.999),
 207 |     eps=1e-06,
 208 |     weight_decay=0.001
 209 | )
 210 | # %%
 211 | if args.resume is not None:
 212 |     checkpoint = torch.load(args.resume)
 213 |     student_model.load_state_dict(checkpoint["model"])
 214 |     optimizer.load_state_dict(checkpoint["optimizer"])
 215 |     best_loss = checkpoint["best_loss"]
 216 |     start_epoch = checkpoint["epoch"] + 1
 217 |     if "logger" in checkpoint:
 218 |         logger.load_checkpoint(checkpoint["logger"])
 219 | else:
 220 |     best_loss = 1e10
 221 |     start_epoch = 0
 222 | # %%
 223 | train_dataset = NpyDataset(data_root=data_root, data_aug=True)
 224 | train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
 225 | # %%
 226 | for epoch in range(start_epoch, max_epochs):
 227 |     epoch_loss = 0
 228 |     logger.log('lrs', optimizer.param_groups[0]['lr'])
 229 |     logger.log('epoch_start_timestamps', time())
 230 |     pbar = tqdm(train_loader)
 231 |     for step, (teacher_input, student_input, image_name) in enumerate(pbar):
 232 |         teacher_input = teacher_input.to(device)
 233 |         student_input = student_input.to(device)
 234 |         with torch.no_grad():
 235 |             teacher_output = teacher_model(teacher_input)
 236 |         student_output = student_model(student_input)
 237 |         target = teacher_output.detach().clone()
 238 |         loss = F.mse_loss(student_output, target)
 239 |         epoch_loss += loss.item()
 240 |         optimizer.zero_grad()
 241 |         loss.backward()
 242 |         optimizer.step()
 243 |         pbar.set_description(f"Epoch {epoch}, loss {loss.item():.4f}")
 244 |     logger.log('epoch_end_timestamps', time())
 245 |     epoch_loss /= (step + 1)
 246 |     logger.log('train_losses', epoch_loss)
 247 |     checkpoint = {
 248 |         "epoch": epoch,
 249 |         "model": student_model.state_dict(),
 250 |         "optimizer": optimizer.state_dict(),
 251 |         "best_loss": best_loss,
 252 |         "logger": logger.get_checkpoint()
 253 |     }
 254 |     torch.save(
 255 |         checkpoint,
 256 |         join(ckpt_dir, f"medsam_lite_encoder_latest.pth")
 257 |     )
 258 |     if epoch_loss < best_loss:
 259 |         best_loss = epoch_loss
 260 |         torch.save(
 261 |             checkpoint,
 262 |             join(ckpt_dir, f"medsam_lite_encoder_best.pth")
 263 |         )
 264 |     logger.plot_progress_png(log_dir)
```
