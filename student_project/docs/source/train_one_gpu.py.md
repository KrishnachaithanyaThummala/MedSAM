# train_one_gpu.py

```python
   1 | # %%
   2 | import os
   3 | import random
   4 | import monai
   5 | from os import listdir, makedirs
   6 | from os.path import join, exists, isfile, isdir, basename
   7 | from glob import glob
   8 | from tqdm import tqdm, trange
   9 | from copy import deepcopy
  10 | from time import time
  11 | import numpy as np
  12 | import torch
  13 | import torch.nn as nn
  14 | import torch.optim as optim
  15 | from torch.utils.data import Dataset, DataLoader
  16 | from datetime import datetime
  17 | 
  18 | from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
  19 | from tiny_vit_sam import TinyViT
  20 | import cv2
  21 | import torch.nn.functional as F
  22 | 
  23 | from matplotlib import pyplot as plt
  24 | import argparse
  25 | # %%
  26 | parser = argparse.ArgumentParser()
  27 | parser.add_argument(
  28 |     "-data_root", type=str, default="./data/npy",
  29 |     help="Path to the npy data root."
  30 | )
  31 | parser.add_argument(
  32 |     "-pretrained_checkpoint", type=str, default="lite_medsam.pth",
  33 |     help="Path to the pretrained Lite-MedSAM checkpoint."
  34 | )
  35 | parser.add_argument(
  36 |     "-resume", type=str, default='workdir/medsam_lite_latest.pth',
  37 |     help="Path to the checkpoint to continue training."
  38 | )
  39 | parser.add_argument(
  40 |     "-work_dir", type=str, default="./workdir",
  41 |     help="Path to the working directory where checkpoints and logs will be saved."
  42 | )
  43 | parser.add_argument(
  44 |     "-num_epochs", type=int, default=10,
  45 |     help="Number of epochs to train."
  46 | )
  47 | parser.add_argument(
  48 |     "-batch_size", type=int, default=4,
  49 |     help="Batch size."
  50 | )
  51 | parser.add_argument(
  52 |     "-num_workers", type=int, default=8,
  53 |     help="Number of workers for dataloader."
  54 | )
  55 | parser.add_argument(
  56 |     "-device", type=str, default="cuda:0",
  57 |     help="Device to train on."
  58 | )
  59 | parser.add_argument(
  60 |     "-bbox_shift", type=int, default=5,
  61 |     help="Perturbation to bounding box coordinates during training."
  62 | )
  63 | parser.add_argument(
  64 |     "-lr", type=float, default=0.00005,
  65 |     help="Learning rate."
  66 | )
  67 | parser.add_argument(
  68 |     "-weight_decay", type=float, default=0.01,
  69 |     help="Weight decay."
  70 | )
  71 | parser.add_argument(
  72 |     "-iou_loss_weight", type=float, default=1.0,
  73 |     help="Weight of IoU loss."
  74 | )
  75 | parser.add_argument(
  76 |     "-seg_loss_weight", type=float, default=1.0,
  77 |     help="Weight of segmentation loss."
  78 | )
  79 | parser.add_argument(
  80 |     "-ce_loss_weight", type=float, default=1.0,
  81 |     help="Weight of cross entropy loss."
  82 | )
  83 | parser.add_argument(
  84 |     "--sanity_check", action="store_true",
  85 |     help="Whether to do sanity check for dataloading."
  86 | )
  87 | 
  88 | args = parser.parse_args()
  89 | # %%
  90 | work_dir = args.work_dir
  91 | data_root = args.data_root
  92 | medsam_lite_checkpoint = args.pretrained_checkpoint
  93 | num_epochs = args.num_epochs
  94 | batch_size = args.batch_size
  95 | num_workers = args.num_workers
  96 | device = args.device
  97 | bbox_shift = args.bbox_shift
  98 | lr = args.lr
  99 | weight_decay = args.weight_decay
 100 | iou_loss_weight = args.iou_loss_weight
 101 | seg_loss_weight = args.seg_loss_weight
 102 | ce_loss_weight = args.ce_loss_weight
 103 | do_sancheck = args.sanity_check
 104 | checkpoint = args.resume
 105 | 
 106 | makedirs(work_dir, exist_ok=True)
 107 | 
 108 | # %%
 109 | torch.cuda.empty_cache()
 110 | os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
 111 | os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
 112 | os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
 113 | os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
 114 | os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
 115 | 
 116 | def show_mask(mask, ax, random_color=False):
 117 |     if random_color:
 118 |         color = np.concatenate([np.random.random(3), np.array([0.45])], axis=0)
 119 |     else:
 120 |         color = np.array([251/255, 252/255, 30/255, 0.45])
 121 |     h, w = mask.shape[-2:]
 122 |     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
 123 |     ax.imshow(mask_image)
 124 |     
 125 | def show_box(box, ax):
 126 |     x0, y0 = box[0], box[1]
 127 |     w, h = box[2] - box[0], box[3] - box[1]
 128 |     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))
 129 | 
 130 | def cal_iou(result, reference):
 131 |     
 132 |     intersection = torch.count_nonzero(torch.logical_and(result, reference), dim=[i for i in range(1, result.ndim)])
 133 |     union = torch.count_nonzero(torch.logical_or(result, reference), dim=[i for i in range(1, result.ndim)])
 134 |     
 135 |     iou = intersection.float() / union.float()
 136 |     
 137 |     return iou.unsqueeze(1)
 138 | 
 139 | # %%
 140 | class NpyDataset(Dataset): 
 141 |     def __init__(self, data_root, image_size=256, bbox_shift=5, data_aug=True):
 142 |         self.data_root = data_root
 143 |         self.gt_path = join(data_root, 'gts')
 144 |         self.img_path = join(data_root, 'imgs')
 145 |         self.gt_path_files = sorted(glob(join(self.gt_path, '*.npy'), recursive=True))
 146 |         self.gt_path_files = [
 147 |             file for file in self.gt_path_files
 148 |             if isfile(join(self.img_path, basename(file)))
 149 |         ]
 150 |         self.image_size = image_size
 151 |         self.target_length = image_size
 152 |         self.bbox_shift = bbox_shift
 153 |         self.data_aug = data_aug
 154 |     
 155 |     def __len__(self):
 156 |         return len(self.gt_path_files)
 157 | 
 158 |     def __getitem__(self, index):
 159 |         img_name = basename(self.gt_path_files[index])
 160 |         assert img_name == basename(self.gt_path_files[index]), 'img gt name error' + self.gt_path_files[index] + self.npy_files[index]
 161 |         img_3c = np.load(join(self.img_path, img_name), 'r', allow_pickle=True) # (H, W, 3)
 162 |         img_resize = self.resize_longest_side(img_3c)
 163 |         # Resizing
 164 |         img_resize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8, a_max=None) # normalize to [0, 1], (H, W, 3
 165 |         img_padded = self.pad_image(img_resize) # (256, 256, 3)
 166 |         # convert the shape to (3, H, W)
 167 |         img_padded = np.transpose(img_padded, (2, 0, 1)) # (3, 256, 256)
 168 |         assert np.max(img_padded)<=1.0 and np.min(img_padded)>=0.0, 'image should be normalized to [0, 1]'
 169 |         gt = np.load(self.gt_path_files[index], 'r', allow_pickle=True) # multiple labels [0, 1,4,5...], (256,256)
 170 |         gt = cv2.resize(
 171 |             gt,
 172 |             (img_resize.shape[1], img_resize.shape[0]),
 173 |             interpolation=cv2.INTER_NEAREST
 174 |         ).astype(np.uint8)
 175 |         gt = self.pad_image(gt) # (256, 256)
 176 |         label_ids = np.unique(gt)[1:]
 177 |         try:
 178 |             gt2D = np.uint8(gt == random.choice(label_ids.tolist())) # only one label, (256, 256)
 179 |         except:
 180 |             print(img_name, 'label_ids.tolist()', label_ids.tolist())
 181 |             gt2D = np.uint8(gt == np.max(gt)) # only one label, (256, 256)
 182 |         # add data augmentation: random fliplr and random flipud
 183 |         if self.data_aug:
 184 |             if random.random() > 0.5:
 185 |                 img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-1))
 186 |                 gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-1))
 187 |                 # print('DA with flip left right')
 188 |             if random.random() > 0.5:
 189 |                 img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-2))
 190 |                 gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-2))
 191 |                 # print('DA with flip upside down')
 192 |         gt2D = np.uint8(gt2D > 0)
 193 |         y_indices, x_indices = np.where(gt2D > 0)
 194 |         x_min, x_max = np.min(x_indices), np.max(x_indices)
 195 |         y_min, y_max = np.min(y_indices), np.max(y_indices)
 196 |         # add perturbation to bounding box coordinates
 197 |         H, W = gt2D.shape
 198 |         x_min = max(0, x_min - random.randint(0, self.bbox_shift))
 199 |         x_max = min(W-1, x_max + random.randint(0, self.bbox_shift))
 200 |         y_min = max(0, y_min - random.randint(0, self.bbox_shift))
 201 |         y_max = min(H-1, y_max + random.randint(0, self.bbox_shift))
 202 |         bboxes = np.array([x_min, y_min, x_max, y_max])
 203 |         return {
 204 |             "image": torch.tensor(img_padded).float(),
 205 |             "gt2D": torch.tensor(gt2D[None, :,:]).long(),
 206 |             "bboxes": torch.tensor(bboxes[None, None, ...]).float(), # (B, 1, 4)
 207 |             "image_name": img_name,
 208 |             "new_size": torch.tensor(np.array([img_resize.shape[0], img_resize.shape[1]])).long(),
 209 |             "original_size": torch.tensor(np.array([img_3c.shape[0], img_3c.shape[1]])).long()
 210 |         }
 211 | 
 212 |     def resize_longest_side(self, image):
 213 |         """
 214 |         Expects a numpy array with shape HxWxC in uint8 format.
 215 |         """
 216 |         long_side_length = self.target_length
 217 |         oldh, oldw = image.shape[0], image.shape[1]
 218 |         scale = long_side_length * 1.0 / max(oldh, oldw)
 219 |         newh, neww = oldh * scale, oldw * scale
 220 |         neww, newh = int(neww + 0.5), int(newh + 0.5)
 221 |         target_size = (neww, newh)
 222 | 
 223 |         return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
 224 | 
 225 |     def pad_image(self, image):
 226 |         """
 227 |         Expects a numpy array with shape HxWxC in uint8 format.
 228 |         """
 229 |         # Pad
 230 |         h, w = image.shape[0], image.shape[1]
 231 |         padh = self.image_size - h
 232 |         padw = self.image_size - w
 233 |         if len(image.shape) == 3: ## Pad image
 234 |             image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
 235 |         else: ## Pad gt mask
 236 |             image_padded = np.pad(image, ((0, padh), (0, padw)))
 237 | 
 238 |         return image_padded
 239 | 
 240 | 
 241 | #%% sanity test of dataset class
 242 | if do_sancheck:
 243 |     tr_dataset = NpyDataset(data_root, data_aug=True)
 244 |     tr_dataloader = DataLoader(tr_dataset, batch_size=8, shuffle=True)
 245 |     for step, batch in enumerate(tr_dataloader):
 246 |         # show the example
 247 |         _, axs = plt.subplots(1, 2, figsize=(10, 10))
 248 |         idx = random.randint(0, 4)
 249 | 
 250 |         image = batch["image"]
 251 |         gt = batch["gt2D"]
 252 |         bboxes = batch["bboxes"]
 253 |         names_temp = batch["image_name"]
 254 | 
 255 |         axs[0].imshow(image[idx].cpu().permute(1,2,0).numpy())
 256 |         show_mask(gt[idx].cpu().squeeze().numpy(), axs[0])
 257 |         show_box(bboxes[idx].numpy().squeeze(), axs[0])
 258 |         axs[0].axis('off')
 259 |         # set title
 260 |         axs[0].set_title(names_temp[idx])
 261 |         idx = random.randint(4, 7)
 262 |         axs[1].imshow(image[idx].cpu().permute(1,2,0).numpy())
 263 |         show_mask(gt[idx].cpu().squeeze().numpy(), axs[1])
 264 |         show_box(bboxes[idx].numpy().squeeze(), axs[1])
 265 |         axs[1].axis('off')
 266 |         # set title
 267 |         axs[1].set_title(names_temp[idx])
 268 |         plt.subplots_adjust(wspace=0.01, hspace=0)
 269 |         plt.savefig(
 270 |             join(work_dir, 'medsam_lite-train_bbox_prompt_sanitycheck_DA.png'),
 271 |             bbox_inches='tight',
 272 |             dpi=300
 273 |         )
 274 |         plt.close()
 275 |         break
 276 | 
 277 | # %%
 278 | class MedSAM_Lite(nn.Module):
 279 |     def __init__(self, 
 280 |                 image_encoder, 
 281 |                 mask_decoder,
 282 |                 prompt_encoder
 283 |                 ):
 284 |         super().__init__()
 285 |         self.image_encoder = image_encoder
 286 |         self.mask_decoder = mask_decoder
 287 |         self.prompt_encoder = prompt_encoder
 288 |         
 289 |     def forward(self, image, boxes):
 290 |         image_embedding = self.image_encoder(image) # (B, 256, 64, 64)
 291 | 
 292 |         sparse_embeddings, dense_embeddings = self.prompt_encoder(
 293 |             points=None,
 294 |             boxes=boxes,
 295 |             masks=None,
 296 |         )
 297 |         low_res_masks, iou_predictions = self.mask_decoder(
 298 |             image_embeddings=image_embedding, # (B, 256, 64, 64)
 299 |             image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
 300 |             sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
 301 |             dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
 302 |             multimask_output=False,
 303 |           ) # (B, 1, 256, 256)
 304 | 
 305 |         return low_res_masks, iou_predictions
 306 | 
 307 |     @torch.no_grad()
 308 |     def postprocess_masks(self, masks, new_size, original_size):
 309 |         """
 310 |         Do cropping and resizing
 311 |         """
 312 |         # Crop
 313 |         masks = masks[:, :, :new_size[0], :new_size[1]]
 314 |         # Resize
 315 |         masks = F.interpolate(
 316 |             masks,
 317 |             size=(original_size[0], original_size[1]),
 318 |             mode="bilinear",
 319 |             align_corners=False,
 320 |         )
 321 | 
 322 |         return masks
 323 | 
 324 | # %%
 325 | medsam_lite_image_encoder = TinyViT(
 326 |     img_size=256,
 327 |     in_chans=3,
 328 |     embed_dims=[
 329 |         64, ## (64, 256, 256)
 330 |         128, ## (128, 128, 128)
 331 |         160, ## (160, 64, 64)
 332 |         320 ## (320, 64, 64) 
 333 |     ],
 334 |     depths=[2, 2, 6, 2],
 335 |     num_heads=[2, 4, 5, 10],
 336 |     window_sizes=[7, 7, 14, 7],
 337 |     mlp_ratio=4.,
 338 |     drop_rate=0.,
 339 |     drop_path_rate=0.0,
 340 |     use_checkpoint=False,
 341 |     mbconv_expand_ratio=4.0,
 342 |     local_conv_size=3,
 343 |     layer_lr_decay=0.8
 344 | )
 345 | 
 346 | medsam_lite_prompt_encoder = PromptEncoder(
 347 |     embed_dim=256,
 348 |     image_embedding_size=(64, 64),
 349 |     input_image_size=(256, 256),
 350 |     mask_in_chans=16
 351 | )
 352 | 
 353 | medsam_lite_mask_decoder = MaskDecoder(
 354 |     num_multimask_outputs=3,
 355 |         transformer=TwoWayTransformer(
 356 |             depth=2,
 357 |             embedding_dim=256,
 358 |             mlp_dim=2048,
 359 |             num_heads=8,
 360 |         ),
 361 |         transformer_dim=256,
 362 |         iou_head_depth=3,
 363 |         iou_head_hidden_dim=256,
 364 | )
 365 | 
 366 | medsam_lite_model = MedSAM_Lite(
 367 |     image_encoder = medsam_lite_image_encoder,
 368 |     mask_decoder = medsam_lite_mask_decoder,
 369 |     prompt_encoder = medsam_lite_prompt_encoder
 370 | )
 371 | 
 372 | if medsam_lite_checkpoint is not None:
 373 |     if isfile(medsam_lite_checkpoint):
 374 |         print(f"Finetuning with pretrained weights {medsam_lite_checkpoint}")
 375 |         medsam_lite_ckpt = torch.load(
 376 |             medsam_lite_checkpoint,
 377 |             map_location="cpu"
 378 |         )
 379 |         medsam_lite_model.load_state_dict(medsam_lite_ckpt, strict=True)
 380 |     else:
 381 |         print(f"Pretained weights {medsam_lite_checkpoint} not found, training from scratch")
 382 | 
 383 | medsam_lite_model = medsam_lite_model.to(device)
 384 | medsam_lite_model.train()
 385 | 
 386 | # %%
 387 | print(f"MedSAM Lite size: {sum(p.numel() for p in medsam_lite_model.parameters())}")
 388 | # %%
 389 | optimizer = optim.AdamW(
 390 |     medsam_lite_model.parameters(),
 391 |     lr=lr,
 392 |     betas=(0.9, 0.999),
 393 |     eps=1e-08,
 394 |     weight_decay=weight_decay,
 395 | )
 396 | lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
 397 |     optimizer,
 398 |     mode='min',
 399 |     factor=0.9,
 400 |     patience=5,
 401 |     cooldown=0
 402 | )
 403 | seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
 404 | ce_loss = nn.BCEWithLogitsLoss(reduction='mean')
 405 | iou_loss = nn.MSELoss(reduction='mean')
 406 | # %%
 407 | train_dataset = NpyDataset(data_root=data_root, data_aug=True)
 408 | train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
 409 | 
 410 | if checkpoint and isfile(checkpoint):
 411 |     print(f"Resuming from checkpoint {checkpoint}")
 412 |     checkpoint = torch.load(checkpoint)
 413 |     medsam_lite_model.load_state_dict(checkpoint["model"], strict=True)
 414 |     optimizer.load_state_dict(checkpoint["optimizer"])
 415 |     start_epoch = checkpoint["epoch"]
 416 |     best_loss = checkpoint["loss"]
 417 |     print(f"Loaded checkpoint from epoch {start_epoch}")
 418 | else:
 419 |     start_epoch = 0
 420 |     best_loss = 1e10
 421 | # %%
 422 | train_losses = []
 423 | for epoch in range(start_epoch + 1, num_epochs):
 424 |     epoch_loss = [1e10 for _ in range(len(train_loader))]
 425 |     epoch_start_time = time()
 426 |     pbar = tqdm(train_loader)
 427 |     for step, batch in enumerate(pbar):
 428 |         image = batch["image"]
 429 |         gt2D = batch["gt2D"]
 430 |         boxes = batch["bboxes"]
 431 |         optimizer.zero_grad()
 432 |         image, gt2D, boxes = image.to(device), gt2D.to(device), boxes.to(device)
 433 |         logits_pred, iou_pred = medsam_lite_model(image, boxes)
 434 |         l_seg = seg_loss(logits_pred, gt2D)
 435 |         l_ce = ce_loss(logits_pred, gt2D.float())
 436 |         #mask_loss = l_seg + l_ce
 437 |         mask_loss = seg_loss_weight * l_seg + ce_loss_weight * l_ce
 438 |         iou_gt = cal_iou(torch.sigmoid(logits_pred) > 0.5, gt2D.bool())
 439 |         l_iou = iou_loss(iou_pred, iou_gt)
 440 |         #loss = mask_loss + l_iou
 441 |         loss = mask_loss + iou_loss_weight * l_iou
 442 |         epoch_loss[step] = loss.item()
 443 |         loss.backward()
 444 |         optimizer.step()
 445 |         optimizer.zero_grad()
 446 |         pbar.set_description(f"Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, loss: {loss.item():.4f}")
 447 | 
 448 |     epoch_end_time = time()
 449 |     epoch_loss_reduced = sum(epoch_loss) / len(epoch_loss)
 450 |     train_losses.append(epoch_loss_reduced)
 451 |     lr_scheduler.step(epoch_loss_reduced)
 452 |     model_weights = medsam_lite_model.state_dict()
 453 |     checkpoint = {
 454 |         "model": model_weights,
 455 |         "epoch": epoch,
 456 |         "optimizer": optimizer.state_dict(),
 457 |         "loss": epoch_loss_reduced,
 458 |         "best_loss": best_loss,
 459 |     }
 460 |     torch.save(checkpoint, join(work_dir, "medsam_lite_latest.pth"))
 461 |     if epoch_loss_reduced < best_loss:
 462 |         print(f"New best loss: {best_loss:.4f} -> {epoch_loss_reduced:.4f}")
 463 |         best_loss = epoch_loss_reduced
 464 |         checkpoint["best_loss"] = best_loss
 465 |         torch.save(checkpoint, join(work_dir, "medsam_lite_best.pth"))
 466 | 
 467 |     epoch_loss_reduced = 1e10
 468 |     # %% plot loss
 469 |     plt.plot(train_losses)
 470 |     plt.title("Dice + Binary Cross Entropy + IoU Loss")
 471 |     plt.xlabel("Epoch")
 472 |     plt.ylabel("Loss")
 473 |     plt.savefig(join(work_dir, "train_loss.png"))
 474 |     plt.close()
```
