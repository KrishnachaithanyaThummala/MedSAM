# distillation/distill_multi_gpu.py

```python
   1 | # %%
   2 | import os
   3 | import glob
   4 | import random
   5 | import shutil
   6 | import argparse
   7 | from os import listdir, makedirs
   8 | from os.path import join, exists, isfile, isdir
   9 | from tqdm import tqdm
  10 | from copy import deepcopy
  11 | from time import time
  12 | from datetime import datetime
  13 | import numpy as np
  14 | import torch
  15 | import torch.nn as nn
  16 | import torch.optim as optim
  17 | from torch.utils.data import Dataset, DataLoader
  18 | from torch import multiprocessing as mp
  19 | from torch.cuda.amp import GradScaler, autocast
  20 | from matplotlib import pyplot as plt
  21 | from segment_anything import sam_model_registry
  22 | import cv2
  23 | import torch.nn.functional as F
  24 | from tiny_vit_sam import TinyViT
  25 | 
  26 | # %%
  27 | torch.cuda.empty_cache()
  28 | torch.set_float32_matmul_precision('high')
  29 | os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
  30 | os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
  31 | os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
  32 | os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
  33 | os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
  34 | 
  35 | # %% Dataset class
  36 | class NpyDataset(Dataset): 
  37 |     def __init__(self, data_root, data_aug=True):
  38 |         self.data_root = data_root
  39 |         self.gt_path = join(data_root, 'gts')
  40 |         self.img_path = join(data_root, 'imgs')
  41 |         self.gt_path_files = sorted(glob.glob(join(self.gt_path, '**/*.npy'), recursive=True))
  42 |         self.gt_path_files = [file for file in self.gt_path_files if os.path.isfile(join(self.img_path, os.path.basename(file)))]
  43 |         self.data_aug = data_aug
  44 |         print(f'number of images: {len(self.gt_path_files)}')
  45 |     
  46 |     def __len__(self):
  47 |         return len(self.gt_path_files)
  48 |     
  49 |     def __getitem__(self, index):
  50 |         img_name = os.path.basename(self.gt_path_files[index])
  51 |         img_3c = np.load(join(self.img_path, img_name), 'r', allow_pickle=True) # (H, W, 3)
  52 |         img_1024 = cv2.resize(
  53 |             img_3c, (1024, 1024), interpolation=cv2.INTER_CUBIC
  54 |         )
  55 |         img_1024 = (img_1024 - img_1024.min()) / np.clip(
  56 |             img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
  57 |         )
  58 |         # convert the shape to (3, H, W)
  59 |         img_1024 = np.transpose(img_1024, (2, 0, 1))
  60 |         assert np.max(img_1024)<=1.0 and np.min(img_1024)>=0.0, 'image should be normalized to [0, 1]'
  61 | 
  62 |         img_256 = cv2.resize(
  63 |             img_3c, (256, 256), interpolation=cv2.INTER_AREA
  64 |         )
  65 |         img_256 = (img_256 - img_256.min()) / np.clip(
  66 |             img_256.max() - img_256.min(), a_min=1e-8, a_max=None
  67 |         )
  68 |         # convert the shape to (3, H, W)
  69 |         img_256 = np.transpose(img_256, (2, 0, 1))
  70 |         assert np.max(img_256)<=1.0 and np.min(img_256)>=0.0, 'image should be normalized to [0, 1]'
  71 | 
  72 |         if self.data_aug:
  73 |             if random.random() > 0.5:
  74 |                 img_1024 = np.ascontiguousarray(np.flip(img_1024, axis=-1))
  75 |                 img_256 = np.ascontiguousarray(np.flip(img_256, axis=-1))
  76 |                 # print('DA with flip left right')
  77 |             if random.random() > 0.5:
  78 |                 img_1024 = np.ascontiguousarray(np.flip(img_1024, axis=-2))
  79 |                 img_256 = np.ascontiguousarray(np.flip(img_256, axis=-2))
  80 |                 # print('DA with flip up down')
  81 | 
  82 |         return torch.tensor(img_1024).float(), torch.tensor(img_256).float(), img_name
  83 | 
  84 | # %%
  85 | class Logger(object):
  86 |     def __init__(self):
  87 |         self.logging = {
  88 |             'train_losses': [],
  89 |             'lrs': [],
  90 |             'epoch_start_timestamps': [],
  91 |             'epoch_end_timestamps': []
  92 |         }
  93 | 
  94 |     def log(self, key, value):
  95 |         self.logging[key].append(value)
  96 |     
  97 |     def plot_progress_png(self, output_folder):
  98 |         fig, ax_all = plt.subplots(2, 1, figsize=(10, 8))
  99 |         x_values = [i for i in range(1, len(self.logging['train_losses']) + 1)]
 100 |         ax_all[0].plot(
 101 |             x_values,
 102 |             self.logging['train_losses'],
 103 |             #color='b',
 104 |             ls='-',
 105 |             label="loss_tr",
 106 |         )
 107 |         ax_all[0].set_ylabel("loss", fontsize=10)
 108 |         ax_all[0].legend(loc=(0, 1))
 109 | 
 110 |         ax_all[1].plot(x_values, [i - j for i, j in zip(self.logging['epoch_end_timestamps'],
 111 |                                                  self.logging['epoch_start_timestamps'])],
 112 |                                                  #color='b',
 113 |                 ls='-',
 114 |                 label="epoch duration",
 115 |                 )
 116 |         ax_all[1].set_xlabel("epoch", fontsize=10)
 117 |         ax_all[1].set_ylabel("time [s]", fontsize=10)
 118 |         ax_all[1].legend(loc=(0, 1))
 119 | 
 120 |         plt.tight_layout()
 121 | 
 122 |         fig.savefig(join(output_folder, "progress.png"))
 123 |         plt.close()
 124 | 
 125 |     def get_checkpoint(self):
 126 |         return self.logging
 127 | 
 128 |     def load_checkpoint(self, checkpoint: dict):
 129 |         self.logging = checkpoint
 130 | 
 131 | # %% set up parser
 132 | def get_args():
 133 |     parser = argparse.ArgumentParser()
 134 |     parser.add_argument('-i', '--tr_npy_path', type=str,
 135 |                         default='',
 136 |                         help='path to training npy files; two subfolders: gts and imgs')
 137 |     parser.add_argument('-task_name', type=str, default='MedSAM-Lite')
 138 |     parser.add_argument('-medsam_checkpoint', type=str, default=None,
 139 |                         help='path to MedSAM encoder checkpoint')
 140 |     parser.add_argument('-tinyvit_checkpoint', type=str, default=None,
 141 |                         help='path to TinyViT encoder checkpoint (not required)')
 142 |     parser.add_argument('-work_dir', type=str, default='./work_dir',
 143 |                         help='path to save the model checkpoints and logs')
 144 |     parser.add_argument('--data_aug', type=bool, default=True, 
 145 |                         help='use data augmentation during training')
 146 |     # train
 147 |     parser.add_argument('-num_epochs', type=int, default=1000)
 148 |     parser.add_argument('-batch_size', type=int, default=8)
 149 |     parser.add_argument('-num_workers', type=int, default=8)
 150 |     # Optimizer parameters
 151 |     parser.add_argument('-weight_decay', type=float, default=0.01,
 152 |                         help='weight decay (default: 0.01)')
 153 |     parser.add_argument('-lr', type=float, default=0.00005, metavar='LR',
 154 |                         help='learning rate (absolute lr)')
 155 |     ## Distributed training args
 156 |     parser.add_argument('-world_size', type=int, help='world size')
 157 |     parser.add_argument('-node_rank', type=int, help='Node rank')
 158 |     parser.add_argument('-bucket_cap_mb', type = int, default = 25,
 159 |                         help='The amount of memory in Mb that DDP will accumulate before firing off gradient communication for the bucket (need to tune)')
 160 |     parser.add_argument('-resume', type = str, default = None,
 161 |                         help="Resuming training from checkpoint folder (only required when resuming training)")
 162 |     parser.add_argument('-init_method', type = str, default = "env://")
 163 | 
 164 |     args = parser.parse_args()
 165 | 
 166 |     return args
 167 | 
 168 | # %%
 169 | def revert_sync_batchnorm(module: torch.nn.Module) -> torch.nn.Module:
 170 |     # Code adapted from https://github.com/pytorch/pytorch/issues/41081#issuecomment-783961547
 171 |     # Original author: Kapil Yedidi (@kapily)
 172 |     converted_module = module
 173 |     if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
 174 |         # Unfortunately, SyncBatchNorm does not store the original class - if it did
 175 |         # we could return the one that was originally created.
 176 |         converted_module = nn.BatchNorm2d(
 177 |             module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats
 178 |         )
 179 |         if module.affine:
 180 |             with torch.no_grad():
 181 |                 converted_module.weight = module.weight
 182 |                 converted_module.bias = module.bias
 183 |         converted_module.running_mean = module.running_mean
 184 |         converted_module.running_var = module.running_var
 185 |         converted_module.num_batches_tracked = module.num_batches_tracked
 186 |         if hasattr(module, "qconfig"):
 187 |             converted_module.qconfig = module.qconfig
 188 |     for name, child in module.named_children():
 189 |         converted_module.add_module(name, revert_sync_batchnorm(child))
 190 |     del module
 191 | 
 192 |     return converted_module
 193 | 
 194 | # %%
 195 | def main(args):
 196 |     ngpus_per_node = torch.cuda.device_count()
 197 |     print("Spwaning processces")
 198 |     mp.spawn(main_worker, nprocs = ngpus_per_node, args=(ngpus_per_node, args))
 199 | 
 200 | 
 201 | def main_worker(gpu, ngpus_per_node, args):
 202 |     node_rank = int(args.node_rank)
 203 |     rank = node_rank * ngpus_per_node + gpu
 204 |     world_size = args.world_size
 205 |     print(f"[Rank {rank}]: Use GPU: {gpu} for training")
 206 |     is_main_host = rank == 0
 207 |     
 208 |     if is_main_host:
 209 |         run_id = datetime.now().strftime("%Y%m%d-%H%M")
 210 |         model_save_path = join(args.work_dir, args.task_name + '-' + run_id)
 211 |         os.makedirs(model_save_path, exist_ok=True)
 212 |         shutil.copyfile(__file__, join(model_save_path, run_id + '_' + os.path.basename(__file__)))
 213 |     torch.cuda.set_device(gpu)
 214 |     torch.distributed.init_process_group(
 215 |         backend = "nccl",
 216 |         init_method = args.init_method,
 217 |         rank = rank,
 218 |         world_size = world_size
 219 |     )
 220 | 
 221 |     medsam_model = sam_model_registry["vit_b"](checkpoint=args.medsam_checkpoint)
 222 |     torch.distributed.barrier()
 223 | 
 224 |     teacher_model = deepcopy(medsam_model.image_encoder)
 225 |     teacher_model.eval()
 226 |     del medsam_model
 227 |     torch.cuda.empty_cache()
 228 |     for param in teacher_model.parameters():
 229 |         param.requires_grad = False ## freeze MedSAM teacher model
 230 |     teacher_model.to(gpu)
 231 | 
 232 |     student_model = TinyViT(
 233 |         img_size=256,
 234 |         in_chans=3,
 235 |         embed_dims=[
 236 |             64,
 237 |             128,
 238 |             160,
 239 |             320
 240 |         ],
 241 |         depths=[2, 2, 6, 2],
 242 |         num_heads=[2, 4, 5, 10],
 243 |         window_sizes=[7, 7, 14, 7],
 244 |         mlp_ratio=4.,
 245 |         drop_rate=0.,
 246 |         drop_path_rate=0.0,
 247 |         use_checkpoint=False,
 248 |         mbconv_expand_ratio=4.0,
 249 |         local_conv_size=3,
 250 |         layer_lr_decay=0.8
 251 |     ) ## TinyViT setup
 252 | 
 253 |     if args.tinyvit_checkpoint is not None:
 254 |         pretrained_student_ckpt = torch.load(
 255 |             args.tinyvit_checkpoint,
 256 |             map_location="cpu"
 257 |         )
 258 |         student_model.load_state_dict(pretrained_student_ckpt)
 259 |         torch.distributed.barrier()
 260 | 
 261 |     student_model.to(gpu)
 262 |     #print("Compiling student model...")
 263 |     #student_model = torch.compile(student_model)
 264 | 
 265 |     for module in student_model.modules():
 266 |         cls_name = module.__class__.__name__
 267 |         if "BatchNorm" in cls_name:
 268 |             assert cls_name == "BatchNorm2d" ## Make sure there's only 2d BN layers, so that I can revert them properly
 269 | 
 270 |     student_model = nn.SyncBatchNorm.convert_sync_batchnorm(student_model)
 271 |     student_model = nn.parallel.DistributedDataParallel(
 272 |         student_model,
 273 |         device_ids = [gpu],
 274 |         output_device = gpu,
 275 |         gradient_as_bucket_view = True,
 276 |         find_unused_parameters = False,
 277 |         bucket_cap_mb = args.bucket_cap_mb
 278 |     )
 279 |     # %%
 280 |     print(f"MedSAM encoder size: {sum(p.numel() for p in teacher_model.parameters())}")
 281 |     print(f"TinyViT encoder size: {sum(p.numel() for p in student_model.parameters())}")
 282 |     # %%
 283 |     optimizer = optim.AdamW(
 284 |         student_model.parameters(),
 285 |         lr=args.lr,
 286 |         betas=(0.9, 0.999),
 287 |         eps=1e-08,
 288 |         weight_decay=args.weight_decay
 289 |     )
 290 | 
 291 |     grad_scaler = GradScaler()
 292 |     logger = Logger()
 293 | 
 294 |     if args.resume is not None:
 295 |         ckpt_folders = sorted(os.listdir(args.resume))
 296 |         ckpt_folders = [f for f in ckpt_folders if (f.startswith(args.task_name) and os.path.isfile(os.path.join(args.resume, f, 'medsam_lite_encoder_latest.pth')))]
 297 |         print('*'*20)
 298 |         print('existing ckpts in', args.resume, ckpt_folders)
 299 |         # find the latest ckpt folders
 300 |         time_strings = [f.split(args.task_name + '-')[-1] for f in ckpt_folders]
 301 |         dates = [datetime.strptime(f, '%Y%m%d-%H%M') for f in time_strings]
 302 |         latest_date = max(dates)
 303 |         latest_ckpt = os.path.join(args.work_dir, args.task_name + '-' + latest_date.strftime('%Y%m%d-%H%M'), 'medsam_lite_encoder_latest.pth')
 304 |         try:
 305 |             ## Map model to be loaded to specified single GPU
 306 |             # loc = 'cuda:{}'.format(gpu)
 307 |             # checkpoint = torch.load(latest_ckpt, map_location = loc)
 308 |             checkpoint = torch.load(latest_ckpt)
 309 |             student_model.module.load_state_dict(checkpoint["model"])
 310 |             optimizer.load_state_dict(checkpoint["optimizer"])
 311 |             best_loss = checkpoint["best_loss"]
 312 |             start_epoch = checkpoint["epoch"] + 1
 313 |             if "logger" in checkpoint:
 314 |                 logger.load_checkpoint(checkpoint["logger"])
 315 |             if "grad_scaler" in checkpoint:
 316 |                 grad_scaler.load_state_dict(checkpoint["grad_scaler"])
 317 | 
 318 |             # medsam_model.load_state_dict(checkpoint['model'])
 319 |             # optimizer.load_state_dict(checkpoint['optimizer'])
 320 |             print(rank, "=> loaded checkpoint '{}' (epoch {})".format(latest_ckpt, checkpoint['epoch']))
 321 |         except:
 322 |             print(rank, "=> no checkpoint found at '{}'".format(latest_ckpt))
 323 |             exit()
 324 |         torch.distributed.barrier()
 325 | 
 326 |     # if args.resume is not None:
 327 |     #     checkpoint = torch.load(args.resume)
 328 |     #     student_model.module.load_state_dict(checkpoint["model"])
 329 |     #     optimizer.load_state_dict(checkpoint["optimizer"])
 330 |     #     best_loss = checkpoint["best_loss"]
 331 |     #     start_epoch = checkpoint["epoch"] + 1
 332 |     #     if "logger" in checkpoint:
 333 |     #         logger.load_checkpoint(checkpoint["logger"])
 334 |     #     if "grad_scaler" in checkpoint:
 335 |     #         grad_scaler.load_state_dict(checkpoint["grad_scaler"])
 336 |     #     torch.distributed.barrier()
 337 |     else:
 338 |         best_loss = 1e10
 339 |         start_epoch = 0
 340 | 
 341 |     # %%
 342 |     train_dataset = NpyDataset(data_root=args.tr_npy_path, data_aug=args.data_aug)
 343 |     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
 344 |     train_loader = DataLoader(
 345 |         train_dataset,
 346 |         batch_size=args.batch_size,
 347 |         shuffle=False,
 348 |         num_workers=args.num_workers,
 349 |         pin_memory=True,
 350 |         sampler=train_sampler
 351 |     )
 352 | 
 353 |     for epoch in range(start_epoch, args.num_epochs):
 354 |         epoch_loss = [None for _ in range(len(train_loader))]
 355 |         logger.log('lrs', optimizer.param_groups[0]['lr'])
 356 |         logger.log('epoch_start_timestamps', time())
 357 |         pbar = tqdm(train_loader)
 358 |         for step, (teacher_input, student_input, image_name) in enumerate(pbar):
 359 |             teacher_input = teacher_input.to(gpu)
 360 |             student_input = student_input.to(gpu)
 361 |             optimizer.zero_grad()
 362 |             with torch.no_grad():
 363 |                 teacher_output = teacher_model(teacher_input)
 364 |             target = teacher_output.detach().clone()
 365 |             with autocast():
 366 |                 student_output = student_model(student_input)
 367 |                 loss = F.mse_loss(student_output, target)
 368 |             epoch_loss[step] = loss.item()
 369 |             grad_scaler.scale(loss).backward()
 370 |             grad_scaler.unscale_(optimizer)
 371 |             grad_scaler.step(optimizer)
 372 |             grad_scaler.update()
 373 |             pbar.set_description(f"[RANK {rank}: GPU {gpu}] Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
 374 |         logger.log('epoch_end_timestamps', time())
 375 |         epoch_loss_world = [None for _ in range(world_size)]
 376 |         torch.distributed.all_gather_object(epoch_loss_world, epoch_loss)
 377 |         epoch_loss_reduced = np.vstack(epoch_loss_world).mean()
 378 |         logger.log('train_losses', epoch_loss_reduced)
 379 |         if is_main_host:
 380 |             module_revert_sync_BN = revert_sync_batchnorm(deepcopy(student_model.module))
 381 |             weights = module_revert_sync_BN.state_dict()
 382 |             checkpoint = {
 383 |                 "epoch": epoch,
 384 |                 "model": weights,
 385 |                 "optimizer": optimizer.state_dict(),
 386 |                 "best_loss": best_loss,
 387 |                 "logger": logger.get_checkpoint(),
 388 |                 "grad_scaler": grad_scaler.state_dict()
 389 |             }
 390 |             torch.save(
 391 |                 checkpoint,
 392 |                 join(model_save_path, "medsam_lite_encoder_latest.pth")
 393 |             )
 394 |             if epoch_loss_reduced < best_loss:
 395 |                 print(f"New best loss: {best_loss:.4f} -> {epoch_loss_reduced:.4f}")
 396 |                 best_loss = epoch_loss_reduced
 397 |                 torch.save(
 398 |                     checkpoint,
 399 |                     join(model_save_path, "medsam_lite_encoder_best.pth")
 400 |                 )
 401 |             logger.plot_progress_png(model_save_path)
 402 |         epoch_loss_reduced = 0
 403 |         torch.distributed.barrier()
 404 | 
 405 | # %%
 406 | if __name__ == "__main__":
 407 |     args = get_args()
 408 |     main(args)
```
