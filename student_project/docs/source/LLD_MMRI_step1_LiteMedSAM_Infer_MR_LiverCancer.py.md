# LLD_MMRI_step1_LiteMedSAM_Infer_MR_LiverCancer.py

```python
   1 | """
   2 | Using MedSAM to generate tumor masks for the LLD_MMRI dataset based on bounding box prompts
   3 | 
   4 | 
   5 | Step 0. Preprocess: run LLD_MMRI_step0_data_preprocess.py
   6 | Step 1. Tumor segmentation with LiteMedSAM
   7 | 
   8 | 
   9 | Jun Ma
  10 | """
  11 | from os import listdir, makedirs
  12 | from os.path import join, isfile, basename
  13 | from glob import glob
  14 | from tqdm import tqdm
  15 | from copy import deepcopy
  16 | from time import time
  17 | import numpy as np
  18 | import torch
  19 | import torch.nn as nn
  20 | import SimpleITK as sitk
  21 | import json
  22 | import torch.nn.functional as F
  23 | 
  24 | from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
  25 | from tiny_vit_sam import TinyViT
  26 | from matplotlib import pyplot as plt
  27 | import cv2
  28 | import torch.multiprocessing as mp
  29 | from skimage import morphology
  30 | 
  31 | 
  32 | #%% set seeds
  33 | torch.set_float32_matmul_precision('high')
  34 | torch.manual_seed(2023)
  35 | torch.cuda.manual_seed(2023)
  36 | np.random.seed(2023)
  37 | 
  38 | #%% set paths
  39 | img_path = "LLD-MMRI/images-nii"
  40 | save_dir = "LLD-MMRI/images-nii-tumor-seg-post"
  41 | json_path = 'LLD-MMRI/labels/Annotation.json'
  42 | medsam_lite_checkpoint_path = 'work_dir/lite_medsam.pth'
  43 | device = 'cuda:0'
  44 | makedirs(save_dir, exist_ok=True)
  45 | 
  46 | with open(json_path) as f:
  47 |     json_data = json.load(f)
  48 | case_names = list(json_data['Annotation_info'].keys())
  49 | 
  50 | bbox_shift = 5
  51 | image_size = 256
  52 | 
  53 | def resize_longest_side(image, target_length):
  54 |     """
  55 |     Expects a numpy array with shape HxWxC in uint8 format.
  56 |     """
  57 |     long_side_length = target_length
  58 |     oldh, oldw = image.shape[0], image.shape[1]
  59 |     scale = long_side_length * 1.0 / max(oldh, oldw)
  60 |     newh, neww = oldh * scale, oldw * scale
  61 |     neww, newh = int(neww + 0.5), int(newh + 0.5)
  62 |     target_size = (neww, newh)
  63 | 
  64 |     return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
  65 | 
  66 | def pad_image(image, target_size):
  67 |     """
  68 |     Expects a numpy array with shape HxWxC in uint8 format.
  69 |     """
  70 |     # Pad
  71 |     h, w = image.shape[0], image.shape[1]
  72 |     padh = target_size - h
  73 |     padw = target_size - w
  74 |     if len(image.shape) == 3: ## Pad image
  75 |         image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
  76 |     else: ## Pad gt mask
  77 |         image_padded = np.pad(image, ((0, padh), (0, padw)))
  78 | 
  79 |     return image_padded
  80 | 
  81 | class MedSAM_Lite(nn.Module):
  82 |     def __init__(
  83 |             self, 
  84 |             image_encoder, 
  85 |             mask_decoder,
  86 |             prompt_encoder
  87 |         ):
  88 |         super().__init__()
  89 |         self.image_encoder = image_encoder
  90 |         self.mask_decoder = mask_decoder
  91 |         self.prompt_encoder = prompt_encoder
  92 | 
  93 |     def forward(self, image, box_np):
  94 |         image_embedding = self.image_encoder(image) # (B, 256, 64, 64)
  95 |         # do not compute gradients for prompt encoder
  96 |         with torch.no_grad():
  97 |             box_torch = torch.as_tensor(box_np, dtype=torch.float32, device=image.device)
  98 |             if len(box_torch.shape) == 2:
  99 |                 box_torch = box_torch[:, None, :] # (B, 1, 4)
 100 | 
 101 |         sparse_embeddings, dense_embeddings = self.prompt_encoder(
 102 |             points=None,
 103 |             boxes=box_np,
 104 |             masks=None,
 105 |         )
 106 |         low_res_masks, iou_predictions = self.mask_decoder(
 107 |             image_embeddings=image_embedding, # (B, 256, 64, 64)
 108 |             image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
 109 |             sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
 110 |             dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
 111 |             multimask_output=False,
 112 |           ) # (B, 1, 256, 256)
 113 | 
 114 |         return low_res_masks
 115 | 
 116 |     @torch.no_grad()
 117 |     def postprocess_masks(self, masks, new_size, original_size):
 118 |         """
 119 |         Do cropping and resizing
 120 | 
 121 |         Parameters
 122 |         ----------
 123 |         masks : torch.Tensor
 124 |             masks predicted by the model
 125 |         new_size : tuple
 126 |             the shape of the image after resizing to the longest side of 256
 127 |         original_size : tuple
 128 |             the original shape of the image
 129 | 
 130 |         Returns
 131 |         -------
 132 |         torch.Tensor
 133 |             the upsampled mask to the original size
 134 |         """
 135 |         # Crop
 136 |         masks = masks[..., :new_size[0], :new_size[1]]
 137 |         # Resize
 138 |         masks = F.interpolate(
 139 |             masks,
 140 |             size=(original_size[0], original_size[1]),
 141 |             mode="bilinear",
 142 |             align_corners=False,
 143 |         )
 144 | 
 145 |         return masks
 146 | 
 147 | 
 148 | def show_mask(mask, ax, mask_color=None, alpha=0.5):
 149 |     if mask_color is not None:
 150 |         color = np.concatenate([mask_color, np.array([alpha])], axis=0)
 151 |     else:
 152 |         color = np.array([251/255, 252/255, 30/255, alpha])
 153 |     h, w = mask.shape[-2:]
 154 |     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
 155 |     ax.imshow(mask_image)
 156 | 
 157 | 
 158 | def show_box(box, ax, edgecolor='blue'):
 159 |     x0, y0 = box[0], box[1]
 160 |     w, h = box[2] - box[0], box[3] - box[1]
 161 |     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))     
 162 | 
 163 | 
 164 | def resize_box(box, new_size, original_size):
 165 |     """
 166 |     Revert box coordinates from scale at 256 to original scale
 167 | 
 168 |     Parameters
 169 |     ----------
 170 |     box : np.ndarray
 171 |         box coordinates at 256 scale
 172 |     new_size : tuple
 173 |         Image shape with the longest edge resized to 256
 174 |     original_size : tuple
 175 |         Original image shape
 176 | 
 177 |     Returns
 178 |     -------
 179 |     np.ndarray
 180 |         box coordinates at original scale
 181 |     """
 182 |     new_box = np.zeros_like(box)
 183 |     ratio = max(original_size) / max(new_size)
 184 |     for i in range(len(box)):
 185 |        new_box[i] = int(box[i] * ratio)
 186 | 
 187 |     return new_box
 188 | 
 189 | 
 190 | @torch.no_grad()
 191 | def medsam_inference(medsam_model, img_embed, box_256, new_size, original_size):
 192 |     box_torch = torch.as_tensor(box_256[None, None, ...], dtype=torch.float, device=img_embed.device)
 193 |     
 194 |     sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
 195 |         points = None,
 196 |         boxes = box_torch,
 197 |         masks = None,
 198 |     )
 199 |     low_res_logits, _ = medsam_model.mask_decoder(
 200 |         image_embeddings=img_embed, # (B, 256, 64, 64)
 201 |         image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
 202 |         sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
 203 |         dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
 204 |         multimask_output=False
 205 |     )
 206 | 
 207 |     low_res_pred = medsam_model.postprocess_masks(low_res_logits, new_size, original_size)
 208 |     low_res_pred = torch.sigmoid(low_res_pred)
 209 |     low_res_pred = low_res_pred.squeeze().cpu().numpy()
 210 |     medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
 211 | 
 212 |     return medsam_seg
 213 | 
 214 | def get_bbox(gt2D, bbox_shift=5):
 215 |     assert np.max(gt2D)==1 and np.min(gt2D)==0.0, f'ground truth should be 0, 1, but got {np.unique(gt2D)}'
 216 |     y_indices, x_indices = np.where(gt2D > 0)
 217 |     x_min, x_max = np.min(x_indices), np.max(x_indices)
 218 |     y_min, y_max = np.min(y_indices), np.max(y_indices)
 219 |     # add perturbation to bounding box coordinates
 220 |     H, W = gt2D.shape
 221 |     x_min = max(0, x_min - bbox_shift)
 222 |     x_max = min(W, x_max + bbox_shift)
 223 |     y_min = max(0, y_min - bbox_shift)
 224 |     y_max = min(H, y_max + bbox_shift)
 225 |     bboxes = np.array([x_min, y_min, x_max, y_max])
 226 | 
 227 |     return bboxes
 228 | 
 229 | 
 230 | medsam_lite_image_encoder = TinyViT(
 231 |     img_size=256,
 232 |     in_chans=3,
 233 |     embed_dims=[
 234 |         64, ## (64, 256, 256)
 235 |         128, ## (128, 128, 128)
 236 |         160, ## (160, 64, 64)
 237 |         320 ## (320, 64, 64) 
 238 |     ],
 239 |     depths=[2, 2, 6, 2],
 240 |     num_heads=[2, 4, 5, 10],
 241 |     window_sizes=[7, 7, 14, 7],
 242 |     mlp_ratio=4.,
 243 |     drop_rate=0.,
 244 |     drop_path_rate=0.0,
 245 |     use_checkpoint=False,
 246 |     mbconv_expand_ratio=4.0,
 247 |     local_conv_size=3,
 248 |     layer_lr_decay=0.8
 249 | )
 250 | 
 251 | medsam_lite_prompt_encoder = PromptEncoder(
 252 |     embed_dim=256,
 253 |     image_embedding_size=(64, 64),
 254 |     input_image_size=(256, 256),
 255 |     mask_in_chans=16
 256 | )
 257 | 
 258 | medsam_lite_mask_decoder = MaskDecoder(
 259 |     num_multimask_outputs=3,
 260 |         transformer=TwoWayTransformer(
 261 |             depth=2,
 262 |             embedding_dim=256,
 263 |             mlp_dim=2048,
 264 |             num_heads=8,
 265 |         ),
 266 |         transformer_dim=256,
 267 |         iou_head_depth=3,
 268 |         iou_head_hidden_dim=256,
 269 | )
 270 | 
 271 | medsam_lite_model = MedSAM_Lite(
 272 |     image_encoder = medsam_lite_image_encoder,
 273 |     mask_decoder = medsam_lite_mask_decoder,
 274 |     prompt_encoder = medsam_lite_prompt_encoder
 275 | )
 276 | 
 277 | medsam_lite_checkpoint = torch.load(medsam_lite_checkpoint_path, map_location='cpu')
 278 | medsam_lite_model.load_state_dict(medsam_lite_checkpoint)
 279 | medsam_lite_model.to(device)
 280 | medsam_lite_model.eval()
 281 | 
 282 | 
 283 | 
 284 | 
 285 | def infer_MR_liver_tumor(case_name):
 286 |     case_phases = json_data['Annotation_info'][case_name]
 287 |     for case_phase in case_phases:
 288 |         phase = case_phase['phase']
 289 |         annotation = case_phase['annotation']
 290 |         num_targets = annotation['num_targets']
 291 |         catetory = annotation['lesion']['0']['category']
 292 |         # load MR data
 293 |         nii_name = case_name + '_' + str(catetory) + '_' + phase.replace(" ","") + '_0000.nii.gz'
 294 |         if num_targets > 1:
 295 |             print(f"Case {case_name} has {num_targets} targets in phase {phase}.")
 296 |         img_sitk = sitk.ReadImage(join(img_path, nii_name))
 297 |         image_data = sitk.GetArrayFromImage(img_sitk) # (D, H, W)
 298 |         
 299 |         # preprocess MR data
 300 |         lower_bound, upper_bound = np.percentile(image_data, 0.5), np.percentile(image_data, 99.5)
 301 |         image_data_pre = np.clip(image_data, lower_bound, upper_bound)
 302 |         image_data_pre = (image_data_pre - np.min(image_data_pre))/(np.max(image_data_pre)-np.min(image_data_pre))*255.0
 303 |         image_data_pre[image_data==0] = 0
 304 |         image_data_pre = np.uint8(image_data_pre)
 305 | 
 306 |         bbox_2d = annotation['lesion']['0']['bbox']['2D_box'] # list; each element is a dict: 'slice_idx', 'x_min', 'y_min', 'x_max', 'y_max', 'area'
 307 |         seg_data_3d = np.zeros_like(image_data, dtype=np.uint8)
 308 |         for bbox in bbox_2d:
 309 |             slice_idx = int(bbox['slice_idx'])
 310 |             img_2d = image_data_pre[slice_idx, :, :]
 311 |             H, W = img_2d.shape
 312 |             img_3c = np.repeat(img_2d[:,:, None], 3, axis=-1)
 313 |             img_256 = resize_longest_side(img_3c, 256)
 314 |             newh, neww = img_256.shape[:2]
 315 |             img_256 = (img_256 - img_256.min()) / np.clip(
 316 |                 img_256.max() - img_256.min(), a_min=1e-8, a_max=None
 317 |             )
 318 |             img_256_padded = pad_image(img_256, 256)
 319 |             img_256_tensor = torch.tensor(img_256_padded).float().permute(2, 0, 1).unsqueeze(0).to(device)
 320 |             with torch.no_grad():
 321 |                 image_embedding = medsam_lite_model.image_encoder(img_256_tensor)
 322 | 
 323 |             xmin, ymin, xmax, ymax = int(bbox['x_min']), int(bbox['y_min']), int(bbox['x_max']), int(bbox['y_max'])
 324 |             xmin = xmin / img_3c.shape[0] * 256
 325 |             xmax = xmax / img_3c.shape[0] * 256
 326 |             ymin = ymin / img_3c.shape[1] * 256
 327 |             ymax = ymax / img_3c.shape[1] * 256
 328 |             bboxes256 = np.array([xmin, ymin, xmax, ymax])
 329 | 
 330 |             seg_mask = medsam_inference(medsam_lite_model, image_embedding, bboxes256, (newh, neww), (H, W))
 331 |             # seg_mask = morphology.remove_small_holes(seg_mask>0)
 332 |             seg_data_3d[slice_idx, :, :] = np.uint8(morphology.remove_small_holes(seg_mask, 150))
 333 | 
 334 | 
 335 |         # save seg_data_3d as nii by sitk
 336 |         seg_data_3d_sitk = sitk.GetImageFromArray(seg_data_3d)
 337 |         seg_data_3d_sitk.SetSpacing(img_sitk.GetSpacing())
 338 |         seg_data_3d_sitk.SetOrigin(img_sitk.GetOrigin())
 339 |         seg_data_3d_sitk.SetDirection(img_sitk.GetDirection())
 340 |         sitk.WriteImage(seg_data_3d_sitk, join(save_dir, nii_name.replace('_0000.nii.gz', '.nii.gz')))
 341 | 
 342 | if __name__ == '__main__':
 343 |     num_workers = 4
 344 |     mp.set_start_method('spawn')
 345 |     with mp.Pool(processes=num_workers) as pool:
 346 |         with tqdm(total=len(case_names)) as pbar:
 347 |             for i, _ in tqdm(enumerate(pool.imap_unordered(infer_MR_liver_tumor, case_names))):
 348 |                 pbar.update()
```
