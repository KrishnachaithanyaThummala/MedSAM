# CVPR24_LiteMedSAM_infer.py

```python
   1 | from os import listdir, makedirs
   2 | from os.path import join, isfile, basename
   3 | from glob import glob
   4 | from tqdm import tqdm
   5 | from time import time
   6 | import numpy as np
   7 | import torch
   8 | import torch.nn as nn
   9 | import torch.nn.functional as F
  10 | 
  11 | from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
  12 | from tiny_vit_sam import TinyViT
  13 | from matplotlib import pyplot as plt
  14 | import cv2
  15 | import argparse
  16 | from collections import OrderedDict
  17 | import pandas as pd
  18 | from datetime import datetime
  19 | 
  20 | 
  21 | #%% set seeds
  22 | torch.set_float32_matmul_precision('high')
  23 | torch.manual_seed(2024)
  24 | torch.cuda.manual_seed(2024)
  25 | np.random.seed(2024)
  26 | 
  27 | parser = argparse.ArgumentParser()
  28 | 
  29 | parser.add_argument(
  30 |     '-i',
  31 |     '--input_dir',
  32 |     type=str,
  33 |     default='test_demo/imgs/',
  34 |     # required=True,
  35 |     help='root directory of the data',
  36 | )
  37 | parser.add_argument(
  38 |     '-o',
  39 |     '--output_dir',
  40 |     type=str,
  41 |     default='test_demo/segs/',
  42 |     help='directory to save the prediction',
  43 | )
  44 | parser.add_argument(
  45 |     '-lite_medsam_checkpoint_path',
  46 |     type=str,
  47 |     default="work_dir/LiteMedSAM/lite_medsam.pth",
  48 |     help='path to the checkpoint of MedSAM-Lite',
  49 | )
  50 | parser.add_argument(
  51 |     '-device',
  52 |     type=str,
  53 |     default="cpu",
  54 |     help='device to run the inference',
  55 | )
  56 | parser.add_argument(
  57 |     '-num_workers',
  58 |     type=int,
  59 |     default=4,
  60 |     help='number of workers for inference with multiprocessing',
  61 | )
  62 | parser.add_argument(
  63 |     '--save_overlay',
  64 |     default=True,
  65 |     action='store_true',
  66 |     help='whether to save the overlay image'
  67 | )
  68 | parser.add_argument(
  69 |     '-png_save_dir',
  70 |     type=str,
  71 |     default='./overlay',
  72 |     help='directory to save the overlay image'
  73 | )
  74 | 
  75 | args = parser.parse_args()
  76 | 
  77 | data_root = args.input_dir
  78 | pred_save_dir = args.output_dir
  79 | save_overlay = args.save_overlay
  80 | num_workers = args.num_workers
  81 | if save_overlay:
  82 |     assert args.png_save_dir is not None, "Please specify the directory to save the overlay image"
  83 |     png_save_dir = args.png_save_dir
  84 |     makedirs(png_save_dir, exist_ok=True)
  85 | 
  86 | lite_medsam_checkpoint_path = args.lite_medsam_checkpoint_path
  87 | makedirs(pred_save_dir, exist_ok=True)
  88 | device = torch.device(args.device)
  89 | image_size = 256
  90 | 
  91 | def resize_longest_side(image, target_length=256):
  92 |     """
  93 |     Resize image to target_length while keeping the aspect ratio
  94 |     Expects a numpy array with shape HxWxC in uint8 format.
  95 |     """
  96 |     oldh, oldw = image.shape[0], image.shape[1]
  97 |     scale = target_length * 1.0 / max(oldh, oldw)
  98 |     newh, neww = oldh * scale, oldw * scale
  99 |     neww, newh = int(neww + 0.5), int(newh + 0.5)
 100 |     target_size = (neww, newh)
 101 | 
 102 |     return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
 103 | 
 104 | def pad_image(image, target_size=256):
 105 |     """
 106 |     Pad image to target_size
 107 |     Expects a numpy array with shape HxWxC in uint8 format.
 108 |     """
 109 |     # Pad
 110 |     h, w = image.shape[0], image.shape[1]
 111 |     padh = target_size - h
 112 |     padw = target_size - w
 113 |     if len(image.shape) == 3: ## Pad image
 114 |         image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
 115 |     else: ## Pad gt mask
 116 |         image_padded = np.pad(image, ((0, padh), (0, padw)))
 117 | 
 118 |     return image_padded
 119 | 
 120 | class MedSAM_Lite(nn.Module):
 121 |     def __init__(
 122 |             self, 
 123 |             image_encoder, 
 124 |             mask_decoder,
 125 |             prompt_encoder
 126 |         ):
 127 |         super().__init__()
 128 |         self.image_encoder = image_encoder
 129 |         self.mask_decoder = mask_decoder
 130 |         self.prompt_encoder = prompt_encoder
 131 | 
 132 |     def forward(self, image, box_np):
 133 |         image_embedding = self.image_encoder(image) # (B, 256, 64, 64)
 134 |         # do not compute gradients for prompt encoder
 135 |         with torch.no_grad():
 136 |             box_torch = torch.as_tensor(box_np, dtype=torch.float32, device=image.device)
 137 |             if len(box_torch.shape) == 2:
 138 |                 box_torch = box_torch[:, None, :] # (B, 1, 4)
 139 | 
 140 |         sparse_embeddings, dense_embeddings = self.prompt_encoder(
 141 |             points=None,
 142 |             boxes=box_np,
 143 |             masks=None,
 144 |         )
 145 |         low_res_masks, iou_predictions = self.mask_decoder(
 146 |             image_embeddings=image_embedding, # (B, 256, 64, 64)
 147 |             image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
 148 |             sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
 149 |             dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
 150 |             multimask_output=False,
 151 |           ) # (B, 1, 256, 256)
 152 | 
 153 |         return low_res_masks
 154 | 
 155 |     @torch.no_grad()
 156 |     def postprocess_masks(self, masks, new_size, original_size):
 157 |         """
 158 |         Do cropping and resizing
 159 | 
 160 |         Parameters
 161 |         ----------
 162 |         masks : torch.Tensor
 163 |             masks predicted by the model
 164 |         new_size : tuple
 165 |             the shape of the image after resizing to the longest side of 256
 166 |         original_size : tuple
 167 |             the original shape of the image
 168 | 
 169 |         Returns
 170 |         -------
 171 |         torch.Tensor
 172 |             the upsampled mask to the original size
 173 |         """
 174 |         # Crop
 175 |         masks = masks[..., :new_size[0], :new_size[1]]
 176 |         # Resize
 177 |         masks = F.interpolate(
 178 |             masks,
 179 |             size=(original_size[0], original_size[1]),
 180 |             mode="bilinear",
 181 |             align_corners=False,
 182 |         )
 183 | 
 184 |         return masks
 185 | 
 186 | 
 187 | def show_mask(mask, ax, mask_color=None, alpha=0.5):
 188 |     """
 189 |     show mask on the image
 190 | 
 191 |     Parameters
 192 |     ----------
 193 |     mask : numpy.ndarray
 194 |         mask of the image
 195 |     ax : matplotlib.axes.Axes
 196 |         axes to plot the mask
 197 |     mask_color : numpy.ndarray
 198 |         color of the mask
 199 |     alpha : float
 200 |         transparency of the mask
 201 |     """
 202 |     if mask_color is not None:
 203 |         color = np.concatenate([mask_color, np.array([alpha])], axis=0)
 204 |     else:
 205 |         color = np.array([251/255, 252/255, 30/255, alpha])
 206 |     h, w = mask.shape[-2:]
 207 |     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
 208 |     ax.imshow(mask_image)
 209 | 
 210 | 
 211 | def show_box(box, ax, edgecolor='blue'):
 212 |     """
 213 |     show bounding box on the image
 214 | 
 215 |     Parameters
 216 |     ----------
 217 |     box : numpy.ndarray
 218 |         bounding box coordinates in the original image
 219 |     ax : matplotlib.axes.Axes
 220 |         axes to plot the bounding box
 221 |     edgecolor : str
 222 |         color of the bounding box
 223 |     """
 224 |     x0, y0 = box[0], box[1]
 225 |     w, h = box[2] - box[0], box[3] - box[1]
 226 |     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))     
 227 | 
 228 | 
 229 | def get_bbox256(mask_256, bbox_shift=3):
 230 |     """
 231 |     Get the bounding box coordinates from the mask (256x256)
 232 | 
 233 |     Parameters
 234 |     ----------
 235 |     mask_256 : numpy.ndarray
 236 |         the mask of the resized image
 237 | 
 238 |     bbox_shift : int
 239 |         Add perturbation to the bounding box coordinates
 240 |     
 241 |     Returns
 242 |     -------
 243 |     numpy.ndarray
 244 |         bounding box coordinates in the resized image
 245 |     """
 246 |     y_indices, x_indices = np.where(mask_256 > 0)
 247 |     x_min, x_max = np.min(x_indices), np.max(x_indices)
 248 |     y_min, y_max = np.min(y_indices), np.max(y_indices)
 249 |     # add perturbation to bounding box coordinates and test the robustness
 250 |     # this can be removed if you do not want to test the robustness
 251 |     H, W = mask_256.shape
 252 |     x_min = max(0, x_min - bbox_shift)
 253 |     x_max = min(W, x_max + bbox_shift)
 254 |     y_min = max(0, y_min - bbox_shift)
 255 |     y_max = min(H, y_max + bbox_shift)
 256 | 
 257 |     bboxes256 = np.array([x_min, y_min, x_max, y_max])
 258 | 
 259 |     return bboxes256
 260 | 
 261 | def resize_box_to_256(box, original_size):
 262 |     """
 263 |     the input bounding box is obtained from the original image
 264 |     here, we rescale it to the coordinates of the resized image
 265 | 
 266 |     Parameters
 267 |     ----------
 268 |     box : numpy.ndarray
 269 |         bounding box coordinates in the original image
 270 |     original_size : tuple
 271 |         the original size of the image
 272 | 
 273 |     Returns
 274 |     -------
 275 |     numpy.ndarray
 276 |         bounding box coordinates in the resized image
 277 |     """
 278 |     new_box = np.zeros_like(box)
 279 |     ratio = 256 / max(original_size)
 280 |     for i in range(len(box)):
 281 |         new_box[i] = int(box[i] * ratio)
 282 | 
 283 |     return new_box
 284 | 
 285 | 
 286 | @torch.no_grad()
 287 | def medsam_inference(medsam_model, img_embed, box_256, new_size, original_size):
 288 |     """
 289 |     Perform inference using the LiteMedSAM model.
 290 | 
 291 |     Args:
 292 |         medsam_model (MedSAMModel): The MedSAM model.
 293 |         img_embed (torch.Tensor): The image embeddings.
 294 |         box_256 (numpy.ndarray): The bounding box coordinates.
 295 |         new_size (tuple): The new size of the image.
 296 |         original_size (tuple): The original size of the image.
 297 |     Returns:
 298 |         tuple: A tuple containing the segmented image and the intersection over union (IoU) score.
 299 |     """
 300 |     box_torch = torch.as_tensor(box_256[None, None, ...], dtype=torch.float, device=img_embed.device)
 301 |     
 302 |     sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
 303 |         points = None,
 304 |         boxes = box_torch,
 305 |         masks = None,
 306 |     )
 307 |     low_res_logits, iou = medsam_model.mask_decoder(
 308 |         image_embeddings=img_embed, # (B, 256, 64, 64)
 309 |         image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
 310 |         sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
 311 |         dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
 312 |         multimask_output=False
 313 |     )
 314 | 
 315 |     low_res_pred = medsam_model.postprocess_masks(low_res_logits, new_size, original_size)
 316 |     low_res_pred = torch.sigmoid(low_res_pred)  
 317 |     low_res_pred = low_res_pred.squeeze().cpu().numpy()  
 318 |     medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
 319 | 
 320 |     return medsam_seg, iou
 321 | 
 322 | medsam_lite_image_encoder = TinyViT(
 323 |     img_size=256,
 324 |     in_chans=3,
 325 |     embed_dims=[
 326 |         64, ## (64, 256, 256)
 327 |         128, ## (128, 128, 128)
 328 |         160, ## (160, 64, 64)
 329 |         320 ## (320, 64, 64) 
 330 |     ],
 331 |     depths=[2, 2, 6, 2],
 332 |     num_heads=[2, 4, 5, 10],
 333 |     window_sizes=[7, 7, 14, 7],
 334 |     mlp_ratio=4.,
 335 |     drop_rate=0.,
 336 |     drop_path_rate=0.0,
 337 |     use_checkpoint=False,
 338 |     mbconv_expand_ratio=4.0,
 339 |     local_conv_size=3,
 340 |     layer_lr_decay=0.8
 341 | )
 342 | 
 343 | medsam_lite_prompt_encoder = PromptEncoder(
 344 |     embed_dim=256,
 345 |     image_embedding_size=(64, 64),
 346 |     input_image_size=(256, 256),
 347 |     mask_in_chans=16
 348 | )
 349 | 
 350 | medsam_lite_mask_decoder = MaskDecoder(
 351 |     num_multimask_outputs=3,
 352 |         transformer=TwoWayTransformer(
 353 |             depth=2,
 354 |             embedding_dim=256,
 355 |             mlp_dim=2048,
 356 |             num_heads=8,
 357 |         ),
 358 |         transformer_dim=256,
 359 |         iou_head_depth=3,
 360 |         iou_head_hidden_dim=256,
 361 | )
 362 | 
 363 | medsam_lite_model = MedSAM_Lite(
 364 |     image_encoder = medsam_lite_image_encoder,
 365 |     mask_decoder = medsam_lite_mask_decoder,
 366 |     prompt_encoder = medsam_lite_prompt_encoder
 367 | )
 368 | 
 369 | lite_medsam_checkpoint = torch.load(lite_medsam_checkpoint_path, map_location='cpu')
 370 | medsam_lite_model.load_state_dict(lite_medsam_checkpoint)
 371 | medsam_lite_model.to(device)
 372 | medsam_lite_model.eval()
 373 | 
 374 | def MedSAM_infer_npz_2D(img_npz_file):
 375 |     npz_name = basename(img_npz_file)
 376 |     npz_data = np.load(img_npz_file, 'r', allow_pickle=True) # (H, W, 3)
 377 |     img_3c = npz_data['imgs'] # (H, W, 3)
 378 |     assert np.max(img_3c)<256, f'input data should be in range [0, 255], but got {np.unique(img_3c)}'
 379 |     H, W = img_3c.shape[:2]
 380 |     boxes = npz_data['boxes']
 381 |     segs = np.zeros(img_3c.shape[:2], dtype=np.uint8)
 382 | 
 383 |     ## preprocessing
 384 |     img_256 = resize_longest_side(img_3c, 256)
 385 |     newh, neww = img_256.shape[:2]
 386 |     img_256_norm = (img_256 - img_256.min()) / np.clip(
 387 |         img_256.max() - img_256.min(), a_min=1e-8, a_max=None
 388 |     )
 389 |     img_256_padded = pad_image(img_256_norm, 256)
 390 |     img_256_tensor = torch.tensor(img_256_padded).float().permute(2, 0, 1).unsqueeze(0).to(device)
 391 |     with torch.no_grad():
 392 |         image_embedding = medsam_lite_model.image_encoder(img_256_tensor)
 393 | 
 394 |     for idx, box in enumerate(boxes, start=1):
 395 |         box256 = resize_box_to_256(box, original_size=(H, W))
 396 |         box256 = box256[None, ...] # (1, 4)
 397 |         medsam_mask, iou_pred = medsam_inference(medsam_lite_model, image_embedding, box256, (newh, neww), (H, W))
 398 |         segs[medsam_mask>0] = idx
 399 |         # print(f'{npz_name}, box: {box}, predicted iou: {np.round(iou_pred.item(), 4)}')
 400 |    
 401 |     np.savez_compressed(
 402 |         join(pred_save_dir, npz_name),
 403 |         segs=segs,
 404 |     )
 405 | 
 406 |     # visualize image, mask and bounding box
 407 |     if save_overlay:
 408 |         fig, ax = plt.subplots(1, 2, figsize=(10, 5))
 409 |         ax[0].imshow(img_3c)
 410 |         ax[1].imshow(img_3c)
 411 |         ax[0].set_title("Image")
 412 |         ax[1].set_title("LiteMedSAM Segmentation")
 413 |         ax[0].axis('off')
 414 |         ax[1].axis('off')
 415 | 
 416 |         for i, box in enumerate(boxes):
 417 |             color = np.random.rand(3)
 418 |             box_viz = box
 419 |             show_box(box_viz, ax[1], edgecolor=color)
 420 |             show_mask((segs == i+1).astype(np.uint8), ax[1], mask_color=color)
 421 | 
 422 |         plt.tight_layout()
 423 |         plt.savefig(join(png_save_dir, npz_name.split(".")[0] + '.png'), dpi=300)
 424 |         plt.close()
 425 | 
 426 | 
 427 | def MedSAM_infer_npz_3D(img_npz_file):
 428 |     npz_name = basename(img_npz_file)
 429 |     npz_data = np.load(img_npz_file, 'r', allow_pickle=True)
 430 |     img_3D = npz_data['imgs'] # (D, H, W)
 431 |     spacing = npz_data['spacing'] # not used in this demo because it treats each slice independently
 432 |     segs = np.zeros_like(img_3D, dtype=np.uint8) 
 433 |     boxes_3D = npz_data['boxes'] # [[x_min, y_min, z_min, x_max, y_max, z_max]]
 434 | 
 435 |     for idx, box3D in enumerate(boxes_3D, start=1):
 436 |         segs_3d_temp = np.zeros_like(img_3D, dtype=np.uint8) 
 437 |         x_min, y_min, z_min, x_max, y_max, z_max = box3D
 438 |         assert z_min < z_max, f"z_min should be smaller than z_max, but got {z_min=} and {z_max=}"
 439 |         mid_slice_bbox_2d = np.array([x_min, y_min, x_max, y_max])
 440 |         z_middle = int((z_max - z_min)/2 + z_min)
 441 | 
 442 |         # infer from middle slice to the z_max
 443 |         # print(npz_name, 'infer from middle slice to the z_max')
 444 |         z_max = min(z_max+1, img_3D.shape[0])
 445 |         for z in range(z_middle, z_max):
 446 |             img_2d = img_3D[z, :, :]
 447 |             if len(img_2d.shape) == 2:
 448 |                 img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
 449 |             else:
 450 |                 img_3c = img_2d
 451 |             H, W, _ = img_3c.shape
 452 | 
 453 |             img_256 = resize_longest_side(img_3c, 256)
 454 |             new_H, new_W = img_256.shape[:2]
 455 | 
 456 |             img_256 = (img_256 - img_256.min()) / np.clip(
 457 |                 img_256.max() - img_256.min(), a_min=1e-8, a_max=None
 458 |             )  # normalize to [0, 1], (H, W, 3)
 459 |             ## Pad image to 256x256
 460 |             img_256 = pad_image(img_256)
 461 |             
 462 |             # convert the shape to (3, H, W)
 463 |             img_256_tensor = torch.tensor(img_256).float().permute(2, 0, 1).unsqueeze(0).to(device)
 464 |             # get the image embedding
 465 |             with torch.no_grad():
 466 |                 image_embedding = medsam_lite_model.image_encoder(img_256_tensor) # (1, 256, 64, 64)
 467 |             if z == z_middle:
 468 |                 box_256 = resize_box_to_256(mid_slice_bbox_2d, original_size=(H, W))
 469 |             else:
 470 |                 pre_seg = segs_3d_temp[z-1, :, :]
 471 |                 pre_seg256 = resize_longest_side(pre_seg)
 472 |                 if np.max(pre_seg256) > 0:
 473 |                     pre_seg256 = pad_image(pre_seg256)
 474 |                     box_256 = get_bbox256(pre_seg256)
 475 |                 else:
 476 |                     box_256 = resize_box_to_256(mid_slice_bbox_2d, original_size=(H, W))
 477 |             img_2d_seg, iou_pred = medsam_inference(medsam_lite_model, image_embedding, box_256, [new_H, new_W], [H, W])
 478 |             segs_3d_temp[z, img_2d_seg>0] = idx
 479 |         
 480 |         # infer from middle slice to the z_max
 481 |         # print(npz_name, 'infer from middle slice to the z_min')
 482 |         z_min = max(-1, z_min-1)
 483 |         for z in range(z_middle-1, z_min, -1):
 484 |             img_2d = img_3D[z, :, :]
 485 |             if len(img_2d.shape) == 2:
 486 |                 img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
 487 |             else:
 488 |                 img_3c = img_2d
 489 |             H, W, _ = img_3c.shape
 490 | 
 491 |             img_256 = resize_longest_side(img_3c)
 492 |             new_H, new_W = img_256.shape[:2]
 493 | 
 494 |             img_256 = (img_256 - img_256.min()) / np.clip(
 495 |                 img_256.max() - img_256.min(), a_min=1e-8, a_max=None
 496 |             )  # normalize to [0, 1], (H, W, 3)
 497 |             ## Pad image to 256x256
 498 |             img_256 = pad_image(img_256)
 499 | 
 500 |             img_256_tensor = torch.tensor(img_256).float().permute(2, 0, 1).unsqueeze(0).to(device)
 501 |             # get the image embedding
 502 |             with torch.no_grad():
 503 |                 image_embedding = medsam_lite_model.image_encoder(img_256_tensor) # (1, 256, 64, 64)
 504 | 
 505 |             pre_seg = segs_3d_temp[z+1, :, :]
 506 |             pre_seg256 = resize_longest_side(pre_seg)
 507 |             if np.max(pre_seg256) > 0:
 508 |                 pre_seg256 = pad_image(pre_seg256)
 509 |                 box_256 = get_bbox256(pre_seg256)
 510 |             else:
 511 |                 box_256 = resize_box_to_256(mid_slice_bbox_2d, original_size=(H, W))
 512 |             img_2d_seg, iou_pred = medsam_inference(medsam_lite_model, image_embedding, box_256, [new_H, new_W], [H, W])
 513 |             segs_3d_temp[z, img_2d_seg>0] = idx
 514 |         segs[segs_3d_temp>0] = idx
 515 |     np.savez_compressed(
 516 |         join(pred_save_dir, npz_name),
 517 |         segs=segs,
 518 |     )            
 519 | 
 520 |     # visualize image, mask and bounding box
 521 |     if save_overlay:
 522 |         idx = int(segs.shape[0] / 2)
 523 |         fig, ax = plt.subplots(1, 2, figsize=(10, 5))
 524 |         ax[0].imshow(img_3D[idx], cmap='gray')
 525 |         ax[1].imshow(img_3D[idx], cmap='gray')
 526 |         ax[0].set_title("Image")
 527 |         ax[1].set_title("LiteMedSAM Segmentation")
 528 |         ax[0].axis('off')
 529 |         ax[1].axis('off')
 530 | 
 531 |         for i, box3D in enumerate(boxes_3D, start=1):
 532 |             if np.sum(segs[idx]==i) > 0:
 533 |                 color = np.random.rand(3)
 534 |                 x_min, y_min, z_min, x_max, y_max, z_max = box3D
 535 |                 box_viz = np.array([x_min, y_min, x_max, y_max])
 536 |                 show_box(box_viz, ax[1], edgecolor=color)
 537 |                 show_mask(segs[idx]==i, ax[1], mask_color=color)
 538 | 
 539 |         plt.tight_layout()
 540 |         plt.savefig(join(png_save_dir, npz_name.split(".")[0] + '.png'), dpi=300)
 541 |         plt.close()
 542 | 
 543 | 
 544 | if __name__ == '__main__':
 545 |     img_npz_files = sorted(glob(join(data_root, '*.npz'), recursive=True))
 546 |     efficiency = OrderedDict()
 547 |     efficiency['case'] = []
 548 |     efficiency['time'] = []
 549 |     for img_npz_file in tqdm(img_npz_files):
 550 |         start_time = time()
 551 |         if basename(img_npz_file).startswith('3D'):
 552 |             MedSAM_infer_npz_3D(img_npz_file)
 553 |         else:
 554 |             MedSAM_infer_npz_2D(img_npz_file)
 555 |         end_time = time()
 556 |         efficiency['case'].append(basename(img_npz_file))
 557 |         efficiency['time'].append(end_time - start_time)
 558 |         current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
 559 |         print(current_time, 'file name:', basename(img_npz_file), 'time cost:', np.round(end_time - start_time, 4))
 560 |     efficiency_df = pd.DataFrame(efficiency)
 561 |     efficiency_df.to_csv(join(pred_save_dir, 'efficiency.csv'), index=False)
```
