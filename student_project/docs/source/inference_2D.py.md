# inference_2D.py

```python
   1 | from glob import glob
   2 | from os import listdir, makedirs
   3 | from os.path import join, isfile, basename
   4 | from tqdm import tqdm, trange
   5 | from copy import deepcopy
   6 | from time import time
   7 | import numpy as np
   8 | import torch
   9 | import torch.nn as nn
  10 | 
  11 | import torch.nn.functional as F
  12 | 
  13 | from tiny_vit_sam import TinyViT
  14 | from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
  15 | from matplotlib import pyplot as plt
  16 | import cv2
  17 | import torch.multiprocessing as mp
  18 | 
  19 | import argparse
  20 | 
  21 | torch.set_float32_matmul_precision('high')
  22 | torch.manual_seed(2024)
  23 | torch.cuda.manual_seed(2024)
  24 | np.random.seed(2024)
  25 | 
  26 | parser = argparse.ArgumentParser()
  27 | 
  28 | parser.add_argument(
  29 |     '-data_root',
  30 |     type=str,
  31 |     default=None,
  32 |     help='root directory of the data',
  33 |     required=True
  34 | )
  35 | parser.add_argument(
  36 |     '-pred_save_dir',
  37 |     type=str,
  38 |     default=None,
  39 |     help='directory to save the prediction',
  40 |     required=True
  41 | )
  42 | parser.add_argument(
  43 |     '-medsam_lite_checkpoint_path',
  44 |     type=str,
  45 |     default="lite_medsam.pth",
  46 |     help='path to the checkpoint of MedSAM-Lite',
  47 |     required=True
  48 | )
  49 | parser.add_argument(
  50 |     '-device',
  51 |     type=str,
  52 |     default="cpu",
  53 |     help='device to run the inference',
  54 | )
  55 | parser.add_argument(
  56 |     '-num_workers',
  57 |     type=int,
  58 |     default=4,
  59 |     help='number of workers for inference with multiprocessing',
  60 | )
  61 | parser.add_argument(
  62 |     '--save_overlay',
  63 |     action='store_true',
  64 |     help='whether to save the overlay image'
  65 | )
  66 | parser.add_argument(
  67 |     '-png_save_dir',
  68 |     type=str,
  69 |     default='./overlay',
  70 |     help='directory to save the overlay image'
  71 | )
  72 | parser.add_argument(
  73 |     '--overwrite',
  74 |     action='store_true',
  75 |     help='whether to overwrite the existing prediction'
  76 | )
  77 | 
  78 | args = parser.parse_args()
  79 | 
  80 | data_root = args.data_root
  81 | pred_save_dir = args.pred_save_dir
  82 | save_overlay = args.save_overlay
  83 | num_workers = args.num_workers
  84 | overwrite = args.overwrite
  85 | medsam_lite_checkpoint_path = args.medsam_lite_checkpoint_path
  86 | if save_overlay:
  87 |     assert args.png_save_dir is not None, "Please specify the directory to save the overlay image"
  88 |     png_save_dir = args.png_save_dir
  89 |     makedirs(png_save_dir, exist_ok=True)
  90 | makedirs(pred_save_dir, exist_ok=True)
  91 | bbox_shift = 5
  92 | device = torch.device(args.device)
  93 | gt_path_files = sorted(glob(join(data_root, '*.npz'), recursive=True))
  94 | image_size = 256
  95 | 
  96 | def resize_longest_side(image, target_length):
  97 |     """
  98 |     Expects a numpy array with shape HxWxC in uint8 format.
  99 |     """
 100 |     long_side_length = target_length
 101 |     oldh, oldw = image.shape[0], image.shape[1]
 102 |     scale = long_side_length * 1.0 / max(oldh, oldw)
 103 |     newh, neww = oldh * scale, oldw * scale
 104 |     neww, newh = int(neww + 0.5), int(newh + 0.5)
 105 |     target_size = (neww, newh)
 106 | 
 107 |     return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
 108 | 
 109 | def pad_image(image, target_size):
 110 |     """
 111 |     Expects a numpy array with shape HxWxC in uint8 format.
 112 |     """
 113 |     # Pad
 114 |     h, w = image.shape[0], image.shape[1]
 115 |     padh = target_size - h
 116 |     padw = target_size - w
 117 |     if len(image.shape) == 3: ## Pad image
 118 |         image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
 119 |     else: ## Pad gt mask
 120 |         image_padded = np.pad(image, ((0, padh), (0, padw)))
 121 | 
 122 |     return image_padded
 123 | 
 124 | # %%
 125 | class MedSAM_Lite(nn.Module):
 126 |     def __init__(
 127 |             self, 
 128 |             image_encoder, 
 129 |             mask_decoder,
 130 |             prompt_encoder
 131 |         ):
 132 |         super().__init__()
 133 |         self.image_encoder = image_encoder
 134 |         self.mask_decoder = mask_decoder
 135 |         self.prompt_encoder = prompt_encoder
 136 | 
 137 |     def forward(self, image, box_np):
 138 |         image_embedding = self.image_encoder(image) # (B, 256, 64, 64)
 139 |         # do not compute gradients for prompt encoder
 140 |         with torch.no_grad():
 141 |             box_torch = torch.as_tensor(box_np, dtype=torch.float32, device=image.device)
 142 |             if len(box_torch.shape) == 2:
 143 |                 box_torch = box_torch[:, None, :] # (B, 1, 4)
 144 | 
 145 |         sparse_embeddings, dense_embeddings = self.prompt_encoder(
 146 |             points=None,
 147 |             boxes=box_np,
 148 |             masks=None,
 149 |         )
 150 |         low_res_masks, iou_predictions = self.mask_decoder(
 151 |             image_embeddings=image_embedding, # (B, 256, 64, 64)
 152 |             image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
 153 |             sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
 154 |             dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
 155 |             multimask_output=False,
 156 |           ) # (B, 1, 256, 256)
 157 | 
 158 |         return low_res_masks
 159 | 
 160 |     @torch.no_grad()
 161 |     def postprocess_masks(self, masks, new_size, original_size):
 162 |         """
 163 |         Do cropping and resizing
 164 | 
 165 |         Parameters
 166 |         ----------
 167 |         masks : torch.Tensor
 168 |             masks predicted by the model
 169 |         new_size : tuple
 170 |             the shape of the image after resizing to the longest side of 256
 171 |         original_size : tuple
 172 |             the original shape of the image
 173 | 
 174 |         Returns
 175 |         -------
 176 |         torch.Tensor
 177 |             the upsampled mask to the original size
 178 |         """
 179 |         # Crop
 180 |         masks = masks[..., :new_size[0], :new_size[1]]
 181 |         # Resize
 182 |         masks = F.interpolate(
 183 |             masks,
 184 |             size=(original_size[0], original_size[1]),
 185 |             mode="bilinear",
 186 |             align_corners=False,
 187 |         )
 188 | 
 189 |         return masks
 190 | 
 191 | 
 192 | def show_mask(mask, ax, mask_color=None, alpha=0.5):
 193 |     if mask_color is not None:
 194 |         color = np.concatenate([mask_color, np.array([alpha])], axis=0)
 195 |     else:
 196 |         color = np.array([251/255, 252/255, 30/255, alpha])
 197 |     h, w = mask.shape[-2:]
 198 |     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
 199 |     ax.imshow(mask_image)
 200 | 
 201 | 
 202 | def show_box(box, ax, edgecolor='blue'):
 203 |     x0, y0 = box[0], box[1]
 204 |     w, h = box[2] - box[0], box[3] - box[1]
 205 |     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))     
 206 | 
 207 | 
 208 | def revert_box(box, new_size, original_size):
 209 |     """
 210 |     Revert box coordinates from scale at 256 to original scale
 211 | 
 212 |     Parameters
 213 |     ----------
 214 |     box : np.ndarray
 215 |         box coordinates at 256 scale
 216 |     new_size : tuple
 217 |         Image shape with the longest edge resized to 256
 218 |     original_size : tuple
 219 |         Original image shape
 220 | 
 221 |     Returns
 222 |     -------
 223 |     np.ndarray
 224 |         box coordinates at original scale
 225 |     """
 226 |     new_box = np.zeros_like(box)
 227 |     ratio = max(original_size) / max(new_size)
 228 |     for i in range(len(box)):
 229 |         new_box[i] = int(box[i] * ratio)
 230 |     
 231 |     return new_box
 232 | 
 233 | 
 234 | @torch.no_grad()
 235 | def medsam_inference(medsam_model, img_embed, box_256, new_size, original_size):
 236 |     box_torch = torch.as_tensor(box_256, dtype=torch.float, device=img_embed.device)
 237 |     if len(box_torch.shape) == 2:
 238 |         box_torch = box_torch[:, None, :] # (B, 1, 4)
 239 |     
 240 |     sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
 241 |         points = None,
 242 |         boxes = box_torch,
 243 |         masks = None,
 244 |     )
 245 |     low_res_logits, _ = medsam_model.mask_decoder(
 246 |         image_embeddings=img_embed, # (B, 256, 64, 64)
 247 |         image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
 248 |         sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
 249 |         dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
 250 |         multimask_output=False
 251 |     )
 252 | 
 253 |     low_res_pred = medsam_model.postprocess_masks(low_res_logits, new_size, original_size)
 254 |     low_res_pred = torch.sigmoid(low_res_pred)
 255 |     low_res_pred = low_res_pred.squeeze().cpu().numpy()
 256 |     medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
 257 | 
 258 |     return medsam_seg
 259 | 
 260 | def get_bbox(gt2D, bbox_shift=5):
 261 |     assert np.max(gt2D)==1 and np.min(gt2D)==0.0, f'ground truth should be 0, 1, but got {np.unique(gt2D)}'
 262 |     y_indices, x_indices = np.where(gt2D > 0)
 263 |     x_min, x_max = np.min(x_indices), np.max(x_indices)
 264 |     y_min, y_max = np.min(y_indices), np.max(y_indices)
 265 |     # add perturbation to bounding box coordinates
 266 |     H, W = gt2D.shape
 267 |     x_min = max(0, x_min - bbox_shift)
 268 |     x_max = min(W, x_max + bbox_shift)
 269 |     y_min = max(0, y_min - bbox_shift)
 270 |     y_max = min(H, y_max + bbox_shift)
 271 |     bboxes = np.array([x_min, y_min, x_max, y_max])
 272 | 
 273 |     return bboxes
 274 | 
 275 | 
 276 | medsam_lite_image_encoder = TinyViT(
 277 |     img_size=256,
 278 |     in_chans=3,
 279 |     embed_dims=[
 280 |         64, ## (64, 256, 256)
 281 |         128, ## (128, 128, 128)
 282 |         160, ## (160, 64, 64)
 283 |         320 ## (320, 64, 64) 
 284 |     ],
 285 |     depths=[2, 2, 6, 2],
 286 |     num_heads=[2, 4, 5, 10],
 287 |     window_sizes=[7, 7, 14, 7],
 288 |     mlp_ratio=4.,
 289 |     drop_rate=0.,
 290 |     drop_path_rate=0.0,
 291 |     use_checkpoint=False,
 292 |     mbconv_expand_ratio=4.0,
 293 |     local_conv_size=3,
 294 |     layer_lr_decay=0.8
 295 | )
 296 | # %%
 297 | medsam_lite_prompt_encoder = PromptEncoder(
 298 |     embed_dim=256,
 299 |     image_embedding_size=(64, 64),
 300 |     input_image_size=(256, 256),
 301 |     mask_in_chans=16
 302 | )
 303 | 
 304 | medsam_lite_mask_decoder = MaskDecoder(
 305 |     num_multimask_outputs=3,
 306 |         transformer=TwoWayTransformer(
 307 |             depth=2,
 308 |             embedding_dim=256,
 309 |             mlp_dim=2048,
 310 |             num_heads=8,
 311 |         ),
 312 |         transformer_dim=256,
 313 |         iou_head_depth=3,
 314 |         iou_head_hidden_dim=256,
 315 | )
 316 | 
 317 | # %%
 318 | medsam_lite_model = MedSAM_Lite(
 319 |     image_encoder = medsam_lite_image_encoder,
 320 |     mask_decoder = medsam_lite_mask_decoder,
 321 |     prompt_encoder = medsam_lite_prompt_encoder
 322 | )
 323 | medsam_lite_checkpoint = torch.load(medsam_lite_checkpoint_path, map_location='cpu')
 324 | medsam_lite_model.load_state_dict(medsam_lite_checkpoint)
 325 | medsam_lite_model.to(device)
 326 | medsam_lite_model.eval()
 327 | 
 328 | # %%
 329 | def MedSAM_infer_npz(gt_path_file):
 330 |     npz_name = basename(gt_path_file)
 331 |     task_folder = gt_path_file.split('/')[-2]
 332 |     makedirs(join(pred_save_dir, task_folder), exist_ok=True)
 333 |     if (not isfile(join(pred_save_dir, task_folder, npz_name))) or overwrite:
 334 |         npz_data = np.load(gt_path_file, 'r', allow_pickle=True) # (H, W, 3)
 335 |         img_3c = npz_data['imgs'] # (H, W, 3)
 336 |         H, W = img_3c.shape[:2]
 337 |         gts = npz_data['gts']
 338 |         if gts.shape != (H, W):
 339 |             gts = cv2.resize(
 340 |                 gts.astype(np.uint8), (W, H),
 341 |                 interpolation=cv2.INTER_NEAREST
 342 |             ).astype(np.uint8)
 343 |         segs = np.zeros(img_3c.shape[:2], dtype=np.uint8)
 344 | 
 345 |         ## MedSAM Lite preprocessing
 346 |         img_256 = resize_longest_side(img_3c, 256)
 347 |         newh, neww = img_256.shape[:2]
 348 |         img_256_norm = (img_256 - img_256.min()) / np.clip(
 349 |             img_256.max() - img_256.min(), a_min=1e-8, a_max=None
 350 |         )
 351 |         img_256_padded = pad_image(img_256_norm, 256)
 352 |         img_256_tensor = torch.tensor(img_256_padded).float().permute(2, 0, 1).unsqueeze(0).to(device)
 353 |         with torch.no_grad():
 354 |             image_embedding = medsam_lite_model.image_encoder(img_256_tensor)
 355 | 
 356 |         label_ids = np.unique(gts)[1:]
 357 |         box_list = []
 358 |         for label_id in label_ids:
 359 |             gt2D = np.uint8(gts == label_id)
 360 |             if gt2D.shape != (newh, neww):
 361 |                 gt2D_resize = cv2.resize(
 362 |                     gt2D.astype(np.uint8), (neww, newh),
 363 |                     interpolation=cv2.INTER_NEAREST
 364 |                 ).astype(np.uint8)
 365 |             else:
 366 |                 gt2D_resize = gt2D.astype(np.uint8)
 367 |             gt2D_padded = pad_image(gt2D_resize, 256)
 368 |             if np.sum(gt2D_padded) > 0:
 369 |                 box = get_bbox(gt2D_padded, bbox_shift) # (4,)
 370 |                 box = box[None, ...] # (1, 4)
 371 |                 sam_mask = medsam_inference(medsam_lite_model, image_embedding, box, (newh, neww), (H, W))
 372 |                 segs[sam_mask>0] = label_id
 373 |                 box_list.append(box.squeeze())
 374 |         
 375 |         label_ids = np.unique(gts)[1:]
 376 | 
 377 |         np.savez_compressed(
 378 |             join(pred_save_dir, task_folder, npz_name),
 379 |             segs=segs, gts=gts
 380 |         )
 381 | 
 382 |         # visualize image, mask and bounding box
 383 |         if save_overlay:
 384 |             fig, ax = plt.subplots(1, 3, figsize=(15, 5))
 385 |             ax[0].imshow(img_3c)
 386 |             ax[1].imshow(img_3c)
 387 |             ax[2].imshow(img_3c)
 388 |             ax[0].set_title("Image")
 389 |             ax[1].set_title("Ground Truth")
 390 |             ax[2].set_title(f"Segmentation")
 391 |             ax[0].axis('off')
 392 |             ax[1].axis('off')
 393 |             ax[2].axis('off')
 394 | 
 395 |             for i, label_id in enumerate(label_ids):
 396 |                 color = np.random.rand(3)
 397 |                 box_viz = revert_box(box_list[i], (newh, neww), (H, W))
 398 |                 show_box(box_viz, ax[1], edgecolor=color)
 399 |                 show_mask((gts == label_id).astype(np.uint8), ax[1], mask_color=color)
 400 |                 show_box(box_viz, ax[2], edgecolor=color)
 401 |                 show_mask((segs == label_id).astype(np.uint8), ax[2], mask_color=color)
 402 | 
 403 |             plt.tight_layout()
 404 |             plt.savefig(join(png_save_dir, npz_name.split(".")[0] + '.png'), dpi=300)
 405 |             plt.close()
 406 | 
 407 | if __name__ == '__main__':
 408 |     num_workers = num_workers
 409 |     mp.set_start_method('spawn')
 410 |     with mp.Pool(processes=num_workers) as pool:
 411 |         with tqdm(total=len(gt_path_files)) as pbar:
 412 |             for i, _ in tqdm(enumerate(pool.imap_unordered(MedSAM_infer_npz, gt_path_files))):
 413 |                 pbar.update()
 414 | 
```
