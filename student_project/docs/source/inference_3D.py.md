# inference_3D.py

```python
   1 | from os import listdir, makedirs
   2 | from os.path import join, isfile, basename
   3 | from glob import glob
   4 | from tqdm import tqdm
   5 | from copy import deepcopy
   6 | from time import time
   7 | import numpy as np
   8 | import torch
   9 | import torch.nn as nn
  10 | 
  11 | import torch.nn.functional as F
  12 | 
  13 | from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
  14 | from tiny_vit_sam import TinyViT
  15 | from matplotlib import pyplot as plt
  16 | import cv2
  17 | import torch.multiprocessing as mp
  18 | 
  19 | import argparse
  20 | 
  21 | #%% set seeds
  22 | torch.set_float32_matmul_precision('high')
  23 | torch.manual_seed(2023)
  24 | torch.cuda.manual_seed(2023)
  25 | np.random.seed(2023)
  26 | 
  27 | parser = argparse.ArgumentParser()
  28 | 
  29 | parser.add_argument(
  30 |     '-data_root',
  31 |     type=str,
  32 |     required=True,
  33 |     help='root directory of the data',
  34 | )
  35 | parser.add_argument(
  36 |     '-pred_save_dir',
  37 |     type=str,
  38 |     required=True,
  39 |     help='directory to save the prediction',
  40 | )
  41 | parser.add_argument(
  42 |     '-medsam_lite_checkpoint_path',
  43 |     type=str,
  44 |     default="workdir/lite_medsam.pth",
  45 |     help='path to the checkpoint of MedSAM-Lite',
  46 | )
  47 | parser.add_argument(
  48 |     '-device',
  49 |     type=str,
  50 |     default="cuda:0",
  51 |     help='device to run the inference',
  52 | )
  53 | parser.add_argument(
  54 |     '-num_workers',
  55 |     type=int,
  56 |     default=4,
  57 |     help='number of workers for inference with multiprocessing',
  58 | )
  59 | parser.add_argument(
  60 |     '--save_overlay',
  61 |     action='store_true',
  62 |     help='whether to save the overlay image'
  63 | )
  64 | parser.add_argument(
  65 |     '-png_save_dir',
  66 |     type=str,
  67 |     default='./overlay/CT_Abd',
  68 |     help='directory to save the overlay image'
  69 | )
  70 | parser.add_argument(
  71 |     '--overwrite',
  72 |     action='store_true',
  73 |     help='whether to overwrite the existing prediction'
  74 | )
  75 | 
  76 | args = parser.parse_args()
  77 | 
  78 | data_root = args.data_root
  79 | pred_save_dir = args.pred_save_dir
  80 | save_overlay = args.save_overlay
  81 | num_workers = args.num_workers
  82 | overwrite = args.overwrite
  83 | if save_overlay:
  84 |     assert args.png_save_dir is not None, "Please specify the directory to save the overlay image"
  85 |     png_save_dir = args.png_save_dir
  86 |     makedirs(png_save_dir, exist_ok=True)
  87 | medsam_lite_checkpoint_path = args.medsam_lite_checkpoint_path
  88 | makedirs(pred_save_dir, exist_ok=True)
  89 | bbox_shift = 5
  90 | device = torch.device(args.device)
  91 | gt_path_files = sorted(glob(join(data_root, '*.npz'), recursive=True))
  92 | image_size = 256
  93 | 
  94 | def resize_longest_side(image, target_length):
  95 |     """
  96 |     Expects a numpy array with shape HxWxC in uint8 format.
  97 |     """
  98 |     long_side_length = target_length
  99 |     oldh, oldw = image.shape[0], image.shape[1]
 100 |     scale = long_side_length * 1.0 / max(oldh, oldw)
 101 |     newh, neww = oldh * scale, oldw * scale
 102 |     neww, newh = int(neww + 0.5), int(newh + 0.5)
 103 |     target_size = (neww, newh)
 104 | 
 105 |     return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
 106 | 
 107 | def pad_image(image, target_size):
 108 |     """
 109 |     Expects a numpy array with shape HxWxC in uint8 format.
 110 |     """
 111 |     # Pad
 112 |     h, w = image.shape[0], image.shape[1]
 113 |     padh = target_size - h
 114 |     padw = target_size - w
 115 |     if len(image.shape) == 3: ## Pad image
 116 |         image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
 117 |     else: ## Pad gt mask
 118 |         image_padded = np.pad(image, ((0, padh), (0, padw)))
 119 | 
 120 |     return image_padded
 121 | 
 122 | class MedSAM_Lite(nn.Module):
 123 |     def __init__(
 124 |             self, 
 125 |             image_encoder, 
 126 |             mask_decoder,
 127 |             prompt_encoder
 128 |         ):
 129 |         super().__init__()
 130 |         self.image_encoder = image_encoder
 131 |         self.mask_decoder = mask_decoder
 132 |         self.prompt_encoder = prompt_encoder
 133 | 
 134 |     def forward(self, image, box_np):
 135 |         image_embedding = self.image_encoder(image) # (B, 256, 64, 64)
 136 |         # do not compute gradients for prompt encoder
 137 |         with torch.no_grad():
 138 |             box_torch = torch.as_tensor(box_np, dtype=torch.float32, device=image.device)
 139 |             if len(box_torch.shape) == 2:
 140 |                 box_torch = box_torch[:, None, :] # (B, 1, 4)
 141 | 
 142 |         sparse_embeddings, dense_embeddings = self.prompt_encoder(
 143 |             points=None,
 144 |             boxes=box_np,
 145 |             masks=None,
 146 |         )
 147 |         low_res_masks, iou_predictions = self.mask_decoder(
 148 |             image_embeddings=image_embedding, # (B, 256, 64, 64)
 149 |             image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
 150 |             sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
 151 |             dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
 152 |             multimask_output=False,
 153 |           ) # (B, 1, 256, 256)
 154 | 
 155 |         return low_res_masks
 156 | 
 157 |     @torch.no_grad()
 158 |     def postprocess_masks(self, masks, new_size, original_size):
 159 |         """
 160 |         Do cropping and resizing
 161 | 
 162 |         Parameters
 163 |         ----------
 164 |         masks : torch.Tensor
 165 |             masks predicted by the model
 166 |         new_size : tuple
 167 |             the shape of the image after resizing to the longest side of 256
 168 |         original_size : tuple
 169 |             the original shape of the image
 170 | 
 171 |         Returns
 172 |         -------
 173 |         torch.Tensor
 174 |             the upsampled mask to the original size
 175 |         """
 176 |         # Crop
 177 |         masks = masks[..., :new_size[0], :new_size[1]]
 178 |         # Resize
 179 |         masks = F.interpolate(
 180 |             masks,
 181 |             size=(original_size[0], original_size[1]),
 182 |             mode="bilinear",
 183 |             align_corners=False,
 184 |         )
 185 | 
 186 |         return masks
 187 | 
 188 | 
 189 | def show_mask(mask, ax, mask_color=None, alpha=0.5):
 190 |     if mask_color is not None:
 191 |         color = np.concatenate([mask_color, np.array([alpha])], axis=0)
 192 |     else:
 193 |         color = np.array([251/255, 252/255, 30/255, alpha])
 194 |     h, w = mask.shape[-2:]
 195 |     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
 196 |     ax.imshow(mask_image)
 197 | 
 198 | 
 199 | def show_box(box, ax, edgecolor='blue'):
 200 |     x0, y0 = box[0], box[1]
 201 |     w, h = box[2] - box[0], box[3] - box[1]
 202 |     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))     
 203 | 
 204 | 
 205 | def resize_box(box, new_size, original_size):
 206 |     """
 207 |     Revert box coordinates from scale at 256 to original scale
 208 | 
 209 |     Parameters
 210 |     ----------
 211 |     box : np.ndarray
 212 |         box coordinates at 256 scale
 213 |     new_size : tuple
 214 |         Image shape with the longest edge resized to 256
 215 |     original_size : tuple
 216 |         Original image shape
 217 | 
 218 |     Returns
 219 |     -------
 220 |     np.ndarray
 221 |         box coordinates at original scale
 222 |     """
 223 |     new_box = np.zeros_like(box)
 224 |     ratio = max(original_size) / max(new_size)
 225 |     for i in range(len(box)):
 226 |        new_box[i] = int(box[i] * ratio)
 227 | 
 228 |     return new_box
 229 | 
 230 | 
 231 | @torch.no_grad()
 232 | def medsam_inference(medsam_model, img_embed, box_256, new_size, original_size):
 233 |     box_torch = torch.as_tensor(box_256[None, None, ...], dtype=torch.float, device=img_embed.device)
 234 |     
 235 |     sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
 236 |         points = None,
 237 |         boxes = box_torch,
 238 |         masks = None,
 239 |     )
 240 |     low_res_logits, _ = medsam_model.mask_decoder(
 241 |         image_embeddings=img_embed, # (B, 256, 64, 64)
 242 |         image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
 243 |         sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
 244 |         dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
 245 |         multimask_output=False
 246 |     )
 247 | 
 248 |     low_res_pred = medsam_model.postprocess_masks(low_res_logits, new_size, original_size)
 249 |     low_res_pred = torch.sigmoid(low_res_pred)
 250 |     low_res_pred = low_res_pred.squeeze().cpu().numpy()
 251 |     medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
 252 | 
 253 |     return medsam_seg
 254 | 
 255 | def get_bbox(gt2D, bbox_shift=5):
 256 |     assert np.max(gt2D)==1 and np.min(gt2D)==0.0, f'ground truth should be 0, 1, but got {np.unique(gt2D)}'
 257 |     y_indices, x_indices = np.where(gt2D > 0)
 258 |     x_min, x_max = np.min(x_indices), np.max(x_indices)
 259 |     y_min, y_max = np.min(y_indices), np.max(y_indices)
 260 |     # add perturbation to bounding box coordinates
 261 |     H, W = gt2D.shape
 262 |     x_min = max(0, x_min - bbox_shift)
 263 |     x_max = min(W, x_max + bbox_shift)
 264 |     y_min = max(0, y_min - bbox_shift)
 265 |     y_max = min(H, y_max + bbox_shift)
 266 |     bboxes = np.array([x_min, y_min, x_max, y_max])
 267 | 
 268 |     return bboxes
 269 | 
 270 | 
 271 | medsam_lite_image_encoder = TinyViT(
 272 |     img_size=256,
 273 |     in_chans=3,
 274 |     embed_dims=[
 275 |         64, ## (64, 256, 256)
 276 |         128, ## (128, 128, 128)
 277 |         160, ## (160, 64, 64)
 278 |         320 ## (320, 64, 64) 
 279 |     ],
 280 |     depths=[2, 2, 6, 2],
 281 |     num_heads=[2, 4, 5, 10],
 282 |     window_sizes=[7, 7, 14, 7],
 283 |     mlp_ratio=4.,
 284 |     drop_rate=0.,
 285 |     drop_path_rate=0.0,
 286 |     use_checkpoint=False,
 287 |     mbconv_expand_ratio=4.0,
 288 |     local_conv_size=3,
 289 |     layer_lr_decay=0.8
 290 | )
 291 | 
 292 | medsam_lite_prompt_encoder = PromptEncoder(
 293 |     embed_dim=256,
 294 |     image_embedding_size=(64, 64),
 295 |     input_image_size=(256, 256),
 296 |     mask_in_chans=16
 297 | )
 298 | 
 299 | medsam_lite_mask_decoder = MaskDecoder(
 300 |     num_multimask_outputs=3,
 301 |         transformer=TwoWayTransformer(
 302 |             depth=2,
 303 |             embedding_dim=256,
 304 |             mlp_dim=2048,
 305 |             num_heads=8,
 306 |         ),
 307 |         transformer_dim=256,
 308 |         iou_head_depth=3,
 309 |         iou_head_hidden_dim=256,
 310 | )
 311 | 
 312 | medsam_lite_model = MedSAM_Lite(
 313 |     image_encoder = medsam_lite_image_encoder,
 314 |     mask_decoder = medsam_lite_mask_decoder,
 315 |     prompt_encoder = medsam_lite_prompt_encoder
 316 | )
 317 | 
 318 | medsam_lite_checkpoint = torch.load(medsam_lite_checkpoint_path, map_location='cpu')
 319 | medsam_lite_model.load_state_dict(medsam_lite_checkpoint)
 320 | medsam_lite_model.to(device)
 321 | medsam_lite_model.eval()
 322 | 
 323 | 
 324 | def MedSAM_infer_npz(gt_path_file):
 325 |     npz_name = basename(gt_path_file)
 326 |     task_folder = gt_path_file.split('/')[-2]
 327 |     makedirs(join(pred_save_dir, task_folder), exist_ok=True)
 328 |     if (not isfile(join(pred_save_dir, task_folder, npz_name))) or overwrite:
 329 |         npz_data = np.load(gt_path_file, 'r', allow_pickle=True) # (H, W, 3)
 330 |         img_3D = npz_data['imgs'] # (Num, H, W)
 331 |         gt_3D = npz_data['gts'] # (Num, H, W)
 332 |         spacing = npz_data['spacing']
 333 |         seg_3D = np.zeros_like(gt_3D, dtype=np.uint8) # (Num, H, W)
 334 |         box_list = [dict() for _ in range(img_3D.shape[0])]
 335 | 
 336 |         for i in range(img_3D.shape[0]):
 337 |             img_2d = img_3D[i,:,:] # (H, W)
 338 |             H, W = img_2d.shape[:2]
 339 |             img_3c = np.repeat(img_2d[:,:, None], 3, axis=-1) # (H, W, 3)
 340 | 
 341 |             ## MedSAM Lite preprocessing
 342 |             img_256 = resize_longest_side(img_3c, 256)
 343 |             newh, neww = img_256.shape[:2]
 344 |             img_256 = (img_256 - img_256.min()) / np.clip(
 345 |                 img_256.max() - img_256.min(), a_min=1e-8, a_max=None
 346 |             )
 347 |             img_256_padded = pad_image(img_256, 256)
 348 |             img_256_tensor = torch.tensor(img_256_padded).float().permute(2, 0, 1).unsqueeze(0).to(device)
 349 |             with torch.no_grad():
 350 |                 image_embedding = medsam_lite_model.image_encoder(img_256_tensor)
 351 | 
 352 |             gt = gt_3D[i,:,:] # (H, W)
 353 |             label_ids = np.unique(gt)[1:]
 354 |             for label_id in label_ids:
 355 |                 gt2D = np.uint8(gt == label_id) # only one label, (H, W)
 356 |                 if gt2D.shape != (newh, neww):
 357 |                     gt2D_resize = cv2.resize(
 358 |                         gt2D.astype(np.uint8), (neww, newh),
 359 |                         interpolation=cv2.INTER_NEAREST
 360 |                     ).astype(np.uint8)
 361 |                 else:
 362 |                     gt2D_resize = gt2D.astype(np.uint8)
 363 |                 gt2D_padded = pad_image(gt2D_resize, 256) ## (256, 256)
 364 |                 if np.sum(gt2D_padded) > 0:
 365 |                     box = get_bbox(gt2D_padded, bbox_shift) # (4,)
 366 |                     sam_mask = medsam_inference(medsam_lite_model, image_embedding, box, (newh, neww), (H, W))
 367 |                     seg_3D[i, sam_mask>0] = label_id
 368 |                     box_list[i][label_id] = box
 369 | 
 370 |         label_ids = np.unique(gt_3D)[1:]
 371 | 
 372 |         np.savez_compressed(
 373 |             join(pred_save_dir, task_folder, npz_name),
 374 |             segs=seg_3D, gts=gt_3D, spacing=spacing
 375 |         )
 376 | 
 377 |         # visualize image, mask and bounding box
 378 |         if save_overlay:
 379 |             idx = int(seg_3D.shape[0] / 2)
 380 |             box_dict = box_list[idx]
 381 |             fig, ax = plt.subplots(1, 3, figsize=(15, 5))
 382 |             ax[0].imshow(img_3D[idx], cmap='gray')
 383 |             ax[1].imshow(img_3D[idx], cmap='gray')
 384 |             ax[2].imshow(img_3D[idx], cmap='gray')
 385 |             ax[0].set_title("Image")
 386 |             ax[1].set_title("Ground Truth")
 387 |             ax[2].set_title(f"Segmentation")
 388 |             ax[0].axis('off')
 389 |             ax[1].axis('off')
 390 |             ax[2].axis('off')
 391 |             for label_id, box_256 in box_dict.items():
 392 |                 color = np.random.rand(3)
 393 |                 box_viz = resize_box(box_256, (newh, neww), (H, W))
 394 |                 show_mask(gt_3D[idx], ax[1], mask_color=color)
 395 |                 show_box(box_viz, ax[1], edgecolor=color)
 396 |                 show_mask(seg_3D[idx], ax[2], mask_color=color)
 397 |                 show_box(box_viz, ax[2], edgecolor=color)
 398 |             plt.tight_layout()
 399 |             plt.savefig(join(png_save_dir, npz_name.split(".")[0] + '.png'), dpi=300)
 400 |             plt.close()
 401 | 
 402 | if __name__ == '__main__':
 403 |     num_workers = num_workers
 404 |     mp.set_start_method('spawn')
 405 |     with mp.Pool(processes=num_workers) as pool:
 406 |         with tqdm(total=len(gt_path_files)) as pbar:
 407 |             for i, _ in tqdm(enumerate(pool.imap_unordered(MedSAM_infer_npz, gt_path_files))):
 408 |                 pbar.update()
```
