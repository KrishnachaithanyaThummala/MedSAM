# train_multi_gpus.py

```python
   1 | # %%
   2 | import os
   3 | import random
   4 | import monai
   5 | from os import listdir, makedirs
   6 | from os.path import join, isfile, basename
   7 | from glob import glob
   8 | from tqdm import tqdm
   9 | from copy import deepcopy
  10 | from time import time
  11 | from shutil import copyfile
  12 | import numpy as np
  13 | import torch
  14 | import torch.nn as nn
  15 | import torch.optim as optim
  16 | from torch.utils.data import Dataset, DataLoader
  17 | from torch import multiprocessing as mp
  18 | from torch import distributed as dist
  19 | from datetime import datetime
  20 | 
  21 | from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
  22 | from tiny_vit_sam import TinyViT
  23 | import cv2
  24 | import torch.nn.functional as F
  25 | 
  26 | from matplotlib import pyplot as plt
  27 | import argparse
  28 | 
  29 | torch.cuda.empty_cache()
  30 | os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
  31 | os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
  32 | os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
  33 | os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
  34 | os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
  35 | 
  36 | 
  37 | def get_args():
  38 |     parser = argparse.ArgumentParser()
  39 |     parser.add_argument('-i', '--tr_npy_path', type=str,
  40 |                         default='data/npy',
  41 |                         help='Path to training npy files; two subfolders: gts and imgs')
  42 |     parser.add_argument('-task_name', type=str, default='MedSAM-Lite')
  43 |     parser.add_argument('-pretrained_checkpoint', type=str, default='lite_medsam.pth',
  44 |                         help='Path to pretrained MedSAM-Lite checkpoint')
  45 |     parser.add_argument('-work_dir', type=str, default='./work_dir')
  46 |     parser.add_argument('--data_aug', action='store_true', default=False,
  47 |                         help='use data augmentation during training')
  48 |     # train
  49 |     parser.add_argument('-num_epochs', type=int, default=1000)
  50 |     parser.add_argument('-batch_size', type=int, default=8)
  51 |     parser.add_argument('-num_workers', type=int, default=8)
  52 |     # Optimizer parameters
  53 |     parser.add_argument('-weight_decay', type=float, default=0.01,
  54 |                         help='weight decay (default: 0.01)')
  55 |     parser.add_argument('-lr', type=float, default=0.0001, metavar='LR',
  56 |                         help='learning rate (absolute lr)')
  57 |     ## Distributed training args
  58 |     parser.add_argument('-world_size', type=int, help='world size')
  59 |     parser.add_argument('-node_rank', type=int, help='Node rank')
  60 |     parser.add_argument('-bucket_cap_mb', type = int, default = 25,
  61 |                         help='The amount of memory in Mb that DDP will accumulate before firing off gradient communication for the bucket (need to tune)')
  62 |     parser.add_argument('-resume', type = str, default = '', required=False,
  63 |                         help="Resuming training from a work_dir")
  64 |     parser.add_argument('-init_method', type = str, default = "env://")
  65 |     args = parser.parse_args()
  66 | 
  67 |     return args
  68 | 
  69 | 
  70 | def show_mask(mask, ax, random_color=False):
  71 |     if random_color:
  72 |         color = np.concatenate([np.random.random(3), np.array([0.45])], axis=0)
  73 |     else:
  74 |         color = np.array([251/255, 252/255, 30/255, 0.45])
  75 |     h, w = mask.shape[-2:]
  76 |     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
  77 |     ax.imshow(mask_image)
  78 |     
  79 | 
  80 | def show_box(box, ax):
  81 |     x0, y0 = box[0], box[1]
  82 |     w, h = box[2] - box[0], box[3] - box[1]
  83 |     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))
  84 | 
  85 | 
  86 | @torch.no_grad()
  87 | def cal_iou(result, reference):
  88 |     
  89 |     intersection = torch.count_nonzero(torch.logical_and(result, reference), dim=[i for i in range(1, result.ndim)])
  90 |     union = torch.count_nonzero(torch.logical_or(result, reference), dim=[i for i in range(1, result.ndim)])
  91 |     
  92 |     iou = intersection.float() / union.float()
  93 |     
  94 |     return iou.unsqueeze(1)
  95 | 
  96 | 
  97 | def revert_sync_batchnorm(module: torch.nn.Module) -> torch.nn.Module:
  98 |     # Code adapted from https://github.com/pytorch/pytorch/issues/41081#issuecomment-783961547
  99 |     # Original author: Kapil Yedidi (@kapily)
 100 |     converted_module = module
 101 |     if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
 102 |         # Unfortunately, SyncBatchNorm does not store the original class - if it did
 103 |         # we could return the one that was originally created.
 104 |         converted_module = nn.BatchNorm2d(
 105 |             module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats
 106 |         )
 107 |         if module.affine:
 108 |             with torch.no_grad():
 109 |                 converted_module.weight = module.weight
 110 |                 converted_module.bias = module.bias
 111 |         converted_module.running_mean = module.running_mean
 112 |         converted_module.running_var = module.running_var
 113 |         converted_module.num_batches_tracked = module.num_batches_tracked
 114 |         if hasattr(module, "qconfig"):
 115 |             converted_module.qconfig = module.qconfig
 116 |     for name, child in module.named_children():
 117 |         converted_module.add_module(name, revert_sync_batchnorm(child))
 118 |     del module
 119 | 
 120 |     return converted_module
 121 | 
 122 | 
 123 | class NpyDataset(Dataset): 
 124 |     def __init__(self, data_root, image_size=256, bbox_shift=10, data_aug=True):
 125 |         self.data_root = data_root
 126 |         self.gt_path = join(data_root, 'gts')
 127 |         self.img_path = join(data_root, 'imgs')
 128 |         self.gt_path_files = sorted(glob(join(self.gt_path, '*.npy'), recursive=True))
 129 |         self.gt_path_files = [file for file in self.gt_path_files if isfile(join(self.img_path, basename(file)))]
 130 |         self.image_size = image_size
 131 |         self.target_length = image_size
 132 |         self.bbox_shift = bbox_shift
 133 |         self.data_aug = data_aug
 134 |     
 135 |     def __len__(self):
 136 |         return len(self.gt_path_files)
 137 | 
 138 |     def __getitem__(self, index):
 139 |         img_name = basename(self.gt_path_files[index])
 140 |         assert img_name == basename(self.gt_path_files[index]), 'img gt name error' + self.gt_path_files[index] + self.npy_files[index]
 141 |         img_3c = np.load(join(self.img_path, img_name), 'r', allow_pickle=True) # (H, W, 3)
 142 |         # Resizing and normalization
 143 |         img_resize = self.resize_longest_side(img_3c)
 144 |         img_resize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8, a_max=None) # normalize to [0, 1], (H, W, 3
 145 |         img_padded = self.pad_image(img_resize) # (256, 256, 3)
 146 |         # convert the shape to (3, H, W)
 147 |         img_padded = np.transpose(img_padded, (2, 0, 1)) # (3, 256, 256)
 148 |         assert np.max(img_padded)<=1.0 and np.min(img_padded)>=0.0, 'image should be normalized to [0, 1]'
 149 |         gt = np.load(self.gt_path_files[index], 'r', allow_pickle=True) # multiple labels [0, 1,4,5...], (256,256)
 150 |         assert gt.max() >= 1, 'gt should have at least one label'
 151 |         gt = cv2.resize(
 152 |             gt,
 153 |             (img_resize.shape[1], img_resize.shape[0]),
 154 |             interpolation=cv2.INTER_NEAREST
 155 |         ).astype(np.uint8)
 156 |         gt = self.pad_image(gt) # (256, 256)
 157 |         label_ids = np.unique(gt)[1:]
 158 |         try:
 159 |             gt2D = np.uint8(gt == random.choice(label_ids.tolist())) # only one label, (256, 256)
 160 |         except:
 161 |             print(img_name, 'label_ids.tolist()', label_ids.tolist())
 162 |             gt2D = np.uint8(gt == np.max(gt)) # only one label, (256, 256)
 163 |         # add data augmentation: random fliplr and random flipud
 164 |         if self.data_aug:
 165 |             if random.random() > 0.5:
 166 |                 img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-1))
 167 |                 gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-1))
 168 |                 # print('DA with flip left right')
 169 |             if random.random() > 0.5:
 170 |                 img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-2))
 171 |                 gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-2))
 172 |                 # print('DA with flip upside down')
 173 |         gt2D = np.uint8(gt2D > 0)
 174 |         y_indices, x_indices = np.where(gt2D > 0)
 175 |         x_min, x_max = np.min(x_indices), np.max(x_indices)
 176 |         y_min, y_max = np.min(y_indices), np.max(y_indices)
 177 |         # add perturbation to bounding box coordinates
 178 |         H, W = gt2D.shape
 179 |         x_min = max(0, x_min - random.randint(0, self.bbox_shift))
 180 |         x_max = min(W-1, x_max + random.randint(0, self.bbox_shift))
 181 |         y_min = max(0, y_min - random.randint(0, self.bbox_shift))
 182 |         y_max = min(H-1, y_max + random.randint(0, self.bbox_shift))
 183 |         bboxes = np.array([x_min, y_min, x_max, y_max])
 184 |         return {
 185 |             "image": torch.tensor(img_padded).float(),
 186 |             "gt2D": torch.tensor(gt2D[None, :,:]).long(),
 187 |             "bboxes": torch.tensor(bboxes[None, None, ...]).float(), # (B, 1, 4)
 188 |             "image_name": img_name,
 189 |             "new_size": torch.tensor(np.array([img_resize.shape[0], img_resize.shape[1]])).long(),
 190 |             "original_size": torch.tensor(np.array([img_3c.shape[0], img_3c.shape[1]])).long()
 191 |         }
 192 | 
 193 |     def resize_longest_side(self, image):
 194 |         """
 195 |         Expects a numpy array with shape HxWxC in uint8 format.
 196 |         """
 197 |         long_side_length = self.target_length
 198 |         oldh, oldw = image.shape[0], image.shape[1]
 199 |         scale = long_side_length * 1.0 / max(oldh, oldw)
 200 |         newh, neww = oldh * scale, oldw * scale
 201 |         neww, newh = int(neww + 0.5), int(newh + 0.5)
 202 |         target_size = (neww, newh)
 203 | 
 204 |         return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
 205 | 
 206 |     def pad_image(self, image):
 207 |         """
 208 |         Expects a numpy array with shape HxWxC in uint8 format.
 209 |         """
 210 |         # Pad
 211 |         h, w = image.shape[0], image.shape[1]
 212 |         padh = self.image_size - h
 213 |         padw = self.image_size - w
 214 |         if len(image.shape) == 3: ## Pad image
 215 |             image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
 216 |         else: ## Pad gt mask
 217 |             image_padded = np.pad(image, ((0, padh), (0, padw)))
 218 | 
 219 |         return image_padded
 220 | 
 221 | def collate_fn(batch):
 222 |     """
 223 |     Collate function for PyTorch DataLoader.
 224 |     """
 225 |     batch_dict = {}
 226 |     for key in batch[0].keys():
 227 |         if key == "image_name":
 228 |             batch_dict[key] = [sample[key] for sample in batch]
 229 |         else:
 230 |             batch_dict[key] = torch.stack([sample[key] for sample in batch], dim=0)
 231 | 
 232 |     return batch_dict
 233 | 
 234 | #%% sanity test of dataset class
 235 | def sanity_check_dataset(args):
 236 |     print('tr_npy_path', args.tr_npy_path)
 237 |     tr_dataset = NpyDataset(args.tr_npy_path, data_aug=args.data_aug)
 238 |     print('len(tr_dataset)', len(tr_dataset))
 239 |     tr_dataloader = DataLoader(tr_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
 240 |     makedirs(args.work_dir, exist_ok=True)
 241 |     for step, batch in enumerate(tr_dataloader):
 242 |         # print(image.shape, gt.shape, bboxes.shape)
 243 |         # show the example
 244 |         _, axs = plt.subplots(1, 2, figsize=(10, 10))
 245 |         idx = random.randint(0, 4)
 246 | 
 247 |         image = batch["image"]
 248 |         gt = batch["gt2D"]
 249 |         bboxes = batch["bboxes"]
 250 |         names_temp = batch["image_name"]
 251 | 
 252 |         axs[0].imshow(image[idx].cpu().permute(1,2,0).numpy())
 253 |         show_mask(gt[idx].cpu().squeeze().numpy(), axs[0])
 254 |         show_box(bboxes[idx].numpy().squeeze(), axs[0])
 255 |         axs[0].axis('off')
 256 |         # set title
 257 |         axs[0].set_title(names_temp[idx])
 258 |         idx = random.randint(4, 7)
 259 |         axs[1].imshow(image[idx].cpu().permute(1,2,0).numpy())
 260 |         show_mask(gt[idx].cpu().squeeze().numpy(), axs[1])
 261 |         show_box(bboxes[idx].numpy().squeeze(), axs[1])
 262 |         axs[1].axis('off')
 263 |         # set title
 264 |         axs[1].set_title(names_temp[idx])
 265 |         # plt.show()  
 266 |         plt.subplots_adjust(wspace=0.01, hspace=0)
 267 |         plt.savefig(
 268 |             join(args.work_dir, 'medsam_lite-train_bbox_prompt_sanitycheck_DA.png'),
 269 |             bbox_inches='tight',
 270 |             dpi=300
 271 |         )
 272 |         plt.close()
 273 |         break
 274 | 
 275 | # %%
 276 | class MedSAM_Lite(nn.Module):
 277 |     def __init__(self, 
 278 |                 image_encoder, 
 279 |                 mask_decoder,
 280 |                 prompt_encoder
 281 |         ):
 282 |         super().__init__()
 283 |         self.image_encoder = image_encoder
 284 |         self.mask_decoder = mask_decoder
 285 |         self.prompt_encoder = prompt_encoder
 286 | 
 287 |     def forward(self, image, boxes):
 288 |         image_embedding = self.image_encoder(image) # (B, 256, 64, 64)
 289 | 
 290 |         sparse_embeddings, dense_embeddings = self.prompt_encoder(
 291 |             points=None,
 292 |             boxes=boxes,
 293 |             masks=None,
 294 |         )
 295 |         low_res_logits, iou_predictions = self.mask_decoder(
 296 |             image_embeddings=image_embedding, # (B, 256, 64, 64)
 297 |             image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
 298 |             sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
 299 |             dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
 300 |             multimask_output=False,
 301 |           ) # (B, 1, 256, 256)
 302 | 
 303 |         return low_res_logits, iou_predictions
 304 | 
 305 |     @torch.no_grad()
 306 |     def postprocess_masks(self, masks, new_size, original_size):
 307 |         """
 308 |         Do cropping and resizing
 309 |         """
 310 |         # Crop
 311 |         masks = masks[:, :, :new_size[0], :new_size[1]]
 312 |         # Resize
 313 |         masks = F.interpolate(
 314 |             masks,
 315 |             size=(original_size[0], original_size[1]),
 316 |             mode="bilinear",
 317 |             align_corners=False,
 318 |         )
 319 | 
 320 |         return masks
 321 | 
 322 | def main(args):
 323 |     ngpus_per_node = torch.cuda.device_count()
 324 |     print("Spwaning processces")
 325 |     mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
 326 | 
 327 | 
 328 | def main_worker(gpu, ngpus_per_node, args):
 329 |     node_rank = int(args.node_rank)
 330 |     rank = node_rank * ngpus_per_node + gpu
 331 |     world_size = args.world_size
 332 |     print(f"[Rank {rank}]: Use GPU: {gpu} for training")
 333 |     is_main_host = rank == 0
 334 |     if is_main_host:
 335 |         run_id = datetime.now().strftime("%Y%m%d-%H%M")
 336 |         model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
 337 |         makedirs(model_save_path, exist_ok=True)
 338 |         copyfile(
 339 |             __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
 340 |         )
 341 |     torch.cuda.set_device(gpu)
 342 |     device = torch.device("cuda:{}".format(gpu))
 343 |     dist.init_process_group(
 344 |         backend="nccl", init_method=args.init_method, rank=rank, world_size=world_size
 345 |     )
 346 | 
 347 |     num_epochs = args.num_epochs
 348 |     batch_size = args.batch_size
 349 |     num_workers = args.num_workers
 350 |     medsam_lite_image_encoder = TinyViT(
 351 |         img_size=256,
 352 |         in_chans=3,
 353 |         embed_dims=[
 354 |             64, ## (64, 256, 256)
 355 |             128, ## (128, 128, 128)
 356 |             160, ## (160, 64, 64)
 357 |             320 ## (320, 64, 64) 
 358 |         ],
 359 |         depths=[2, 2, 6, 2],
 360 |         num_heads=[2, 4, 5, 10],
 361 |         window_sizes=[7, 7, 14, 7],
 362 |         mlp_ratio=4.,
 363 |         drop_rate=0.,
 364 |         drop_path_rate=0.0,
 365 |         use_checkpoint=False,
 366 |         mbconv_expand_ratio=4.0,
 367 |         local_conv_size=3,
 368 |         layer_lr_decay=0.8
 369 |     )
 370 | 
 371 |     medsam_lite_prompt_encoder = PromptEncoder(
 372 |         embed_dim=256,
 373 |         image_embedding_size=(64, 64),
 374 |         input_image_size=(256, 256),
 375 |         mask_in_chans=16
 376 |     )
 377 | 
 378 |     medsam_lite_mask_decoder = MaskDecoder(
 379 |         num_multimask_outputs=3,
 380 |             transformer=TwoWayTransformer(
 381 |                 depth=2,
 382 |                 embedding_dim=256,
 383 |                 mlp_dim=2048,
 384 |                 num_heads=8,
 385 |             ),
 386 |             transformer_dim=256,
 387 |             iou_head_depth=3,
 388 |             iou_head_hidden_dim=256,
 389 |     )
 390 |     
 391 |     medsam_lite_model = MedSAM_Lite(
 392 |         image_encoder = medsam_lite_image_encoder,
 393 |         mask_decoder = medsam_lite_mask_decoder,
 394 |         prompt_encoder = medsam_lite_prompt_encoder
 395 |     )
 396 |     
 397 |     if (not os.path.exists(args.resume)) and isfile(args.pretrained_checkpoint):
 398 |         ## Load pretrained checkpoint if there's no checkpoint to resume from and there's a pretrained checkpoint
 399 |         print(f"Loading pretrained checkpoint from {args.pretrained_checkpoint}")
 400 |         medsam_lite_checkpoint = torch.load(args.pretrained_checkpoint, map_location="cpu")
 401 |         medsam_lite_model.load_state_dict(medsam_lite_checkpoint, strict=True)
 402 | 
 403 |     medsam_lite_model = medsam_lite_model.to(device)
 404 | 
 405 |     ## Make sure there's only 2d BN layers, so that I can revert them properly
 406 |     for module in medsam_lite_model.modules():
 407 |         cls_name = module.__class__.__name__
 408 |         if "BatchNorm" in cls_name:
 409 |             assert cls_name == "BatchNorm2d" 
 410 |     medsam_lite_model = nn.SyncBatchNorm.convert_sync_batchnorm(medsam_lite_model)
 411 | 
 412 |     medsam_lite_model = nn.parallel.DistributedDataParallel(
 413 |         medsam_lite_model,
 414 |         device_ids=[gpu],
 415 |         output_device=gpu,
 416 |         find_unused_parameters=True,
 417 |         bucket_cap_mb=args.bucket_cap_mb
 418 |     )
 419 |     medsam_lite_model.train()
 420 |     # %%
 421 |     print(f"MedSAM Lite size: {sum(p.numel() for p in medsam_lite_model.parameters())}")
 422 |     # %%
 423 |     optimizer = optim.AdamW(
 424 |         medsam_lite_model.parameters(),
 425 |         lr=args.lr,
 426 |         betas=(0.9, 0.999),
 427 |         eps=1e-08,
 428 |         weight_decay=args.weight_decay,
 429 |     )
 430 |     lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
 431 |         optimizer,
 432 |         mode='min',
 433 |         factor=0.9,
 434 |         patience=5,
 435 |         cooldown=0
 436 |     )
 437 |     seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
 438 |     ce_loss = nn.BCEWithLogitsLoss(reduction='mean')
 439 |     iou_loss = nn.MSELoss(reduction='mean')
 440 |     # %%
 441 |     data_root = args.tr_npy_path
 442 |     train_dataset = NpyDataset(data_root=data_root, data_aug=True)
 443 |     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
 444 |     train_loader = DataLoader(
 445 |         train_dataset,
 446 |         batch_size=batch_size,
 447 |         shuffle=False,
 448 |         num_workers=num_workers,
 449 |         pin_memory=True,
 450 |         sampler=train_sampler,
 451 |         collate_fn=collate_fn
 452 |     )
 453 |     # %%
 454 | 
 455 |     if os.path.exists(args.resume):
 456 |         ckpt_folders = sorted(listdir(args.resume))
 457 |         ckpt_folders = [f for f in ckpt_folders if (f.startswith(args.task_name) and isfile(join(args.resume, f, 'medsam_lite_latest.pth')))]
 458 |         print('*'*20)
 459 |         print('existing ckpts in', args.resume, ckpt_folders)
 460 |         # find the latest ckpt folders
 461 |         time_strings = [f.split(args.task_name + '-')[-1] for f in ckpt_folders]
 462 |         dates = [datetime.strptime(f, '%Y%m%d-%H%M') for f in time_strings]
 463 |         latest_date = max(dates)
 464 |         latest_ckpt = join(args.work_dir, args.task_name + '-' + latest_date.strftime('%Y%m%d-%H%M'), 'medsam_lite_latest.pth')
 465 |         print('Loading from', latest_ckpt)
 466 |         checkpoint = torch.load(latest_ckpt, map_location=device)
 467 |         medsam_lite_model.module.load_state_dict(checkpoint["model"])
 468 |         optimizer.load_state_dict(checkpoint["optimizer"])
 469 |         start_epoch = checkpoint["epoch"] + 1
 470 |         best_loss = checkpoint["loss"]
 471 |         print(f"Loaded checkpoint from epoch {start_epoch}")
 472 |     else:
 473 |         start_epoch = 0
 474 |         best_loss = 1e10
 475 | 
 476 |     train_losses = []
 477 |     epoch_times = []
 478 |     for epoch in range(start_epoch, num_epochs):
 479 |         epoch_loss = [1e10 for _ in range(len(train_loader))]
 480 |         epoch_start_time = time()
 481 |         pbar = tqdm(train_loader)
 482 |         for step, batch in enumerate(pbar):
 483 |             image = batch["image"]
 484 |             gt2D = batch["gt2D"]
 485 |             boxes = batch["bboxes"]
 486 |             optimizer.zero_grad()
 487 |             image, gt2D, boxes = image.to(device), gt2D.to(device), boxes.to(device)
 488 |             logits_pred, iou_pred = medsam_lite_model(image, boxes)
 489 |             l_seg = seg_loss(logits_pred, gt2D)
 490 |             l_ce = ce_loss(logits_pred, gt2D.float())
 491 |             mask_loss = l_seg + l_ce
 492 |             with torch.no_grad():
 493 |                 iou_gt = cal_iou(torch.sigmoid(logits_pred) > 0.5, gt2D.bool())
 494 |             l_iou = iou_loss(iou_pred, iou_gt)
 495 |             loss = mask_loss + l_iou
 496 |             epoch_loss[step] = loss.item()
 497 |             loss.backward()
 498 |             optimizer.step()
 499 |             optimizer.zero_grad()
 500 |             pbar.set_description(f"[RANK {rank}] Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, loss: {loss.item():.4f}")
 501 | 
 502 |         epoch_end_time = time()
 503 |         epoch_duration = epoch_end_time - epoch_start_time
 504 |         epoch_times.append(epoch_duration)
 505 |         epoch_loss_world = [None for _ in range(world_size)]
 506 |         dist.all_gather_object(epoch_loss_world, epoch_loss)
 507 |         epoch_loss_reduced = np.vstack(epoch_loss_world).mean()
 508 |         train_losses.append(epoch_loss_reduced)
 509 |         lr_scheduler.step(epoch_loss_reduced)
 510 | 
 511 |         if is_main_host:
 512 |             module_revert_sync_BN = revert_sync_batchnorm(deepcopy(medsam_lite_model.module))
 513 |             weights = module_revert_sync_BN.state_dict()
 514 |             checkpoint = {
 515 |                 "model": weights,
 516 |                 "epoch": epoch,
 517 |                 "optimizer": optimizer.state_dict(),
 518 |                 "loss": epoch_loss_reduced,
 519 |                 "best_loss": best_loss,
 520 |             }
 521 |             torch.save(checkpoint, join(model_save_path, "medsam_lite_latest.pth"))
 522 |         if epoch_loss_reduced < best_loss:
 523 |             print(f"New best loss: {best_loss:.4f} -> {epoch_loss_reduced:.4f}")
 524 |             best_loss = epoch_loss_reduced
 525 |             if is_main_host:
 526 |                 checkpoint["best_loss"] = best_loss
 527 |                 torch.save(checkpoint, join(model_save_path, "medsam_lite_best.pth"))
 528 |         dist.barrier()
 529 |         epoch_loss_reduced = 1e10
 530 |         # %% plot loss
 531 |         if is_main_host:
 532 |             fig, axes = plt.subplots(2, 1, figsize=(10, 8))
 533 |             axes[0].title.set_text("Dice + Binary Cross Entropy + IoU Loss")
 534 |             axes[0].plot(train_losses)
 535 |             axes[0].set_ylabel("Loss")
 536 |             axes[1].plot(epoch_times)
 537 |             axes[1].title.set_text("Epoch Duration")
 538 |             axes[1].set_ylabel("Duration (s)")
 539 |             axes[1].set_xlabel("Epoch")
 540 |             plt.tight_layout()
 541 |             plt.savefig(join(model_save_path, "log.png"))
 542 |             plt.close()
 543 |         dist.barrier()
 544 | 
 545 |         
 546 | # %%
 547 | if __name__ == "__main__":
 548 |     args = get_args()
 549 |     sanity_check_dataset(args)
 550 |     main(args)
```
