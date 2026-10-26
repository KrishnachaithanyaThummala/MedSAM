# utils/demo.py

```python
   1 | import gc
   2 | import torch
   3 | import numpy as np
   4 | import cv2
   5 | from torch.nn import functional as F
   6 | from os import listdir, makedirs, getcwd
   7 | from os.path import join, exists, isfile, isdir, basename
   8 | from glob import glob
   9 | from ipywidgets import interact, widgets, FileUpload
  10 | from IPython.display import display
  11 | from matplotlib import patches as patches
  12 | from matplotlib import pyplot as plt
  13 | from copy import deepcopy
  14 | 
  15 | def show_mask(mask, ax, random_color=False, alpha=0.95):
  16 |     if random_color:
  17 |         color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
  18 |     else:
  19 |         color = np.array([251/255, 252/255, 30/255, alpha])
  20 |     h, w = mask.shape[-2:]
  21 |     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
  22 |     ax.imshow(mask_image)
  23 | 
  24 | 
  25 | class BboxPromptDemo:
  26 |     def __init__(self, model):
  27 |         self.model = model
  28 |         self.model.eval()
  29 |         self.image = None
  30 |         self.image_embeddings = None
  31 |         self.img_size = None
  32 |         self.gt = None
  33 |         self.currently_selecting = False
  34 |         self.x0, self.y0, self.x1, self.y1 = 0., 0., 0., 0.
  35 |         self.rect = None
  36 |         self.fig, self.axes = None, None
  37 |         self.segs = []
  38 | 
  39 |     def _show(self, fig_size=5, random_color=True, alpha=0.65):
  40 |         assert self.image is not None, "Please set image first."
  41 | 
  42 |         self.fig, self.axes = plt.subplots(1, 1, figsize=(fig_size, fig_size))
  43 |         self.fig.canvas.header_visible = False
  44 |         self.fig.canvas.footer_visible = False
  45 |         self.fig.canvas.toolbar_visible = False
  46 |         self.fig.canvas.resizable = False
  47 | 
  48 |         plt.tight_layout()
  49 |         self.axes.imshow(self.image)
  50 |         self.axes.axis('off')
  51 | 
  52 |         def __on_press(event):
  53 |             if event.inaxes == self.axes:
  54 |                 self.x0 = float(event.xdata) 
  55 |                 self.y0 = float(event.ydata)
  56 |                 self.currently_selecting = True
  57 |                 self.rect = plt.Rectangle(
  58 |                     (self.x0, self.y0),
  59 |                     1,1, linestyle="--",
  60 |                     edgecolor="crimson", fill=False
  61 |                 )
  62 |                 self.axes.add_patch(self.rect)
  63 |                 self.rect.set_visible(False)
  64 | 
  65 |         def __on_release(event):
  66 |             if event.inaxes == self.axes:
  67 |                 if self.currently_selecting:
  68 |                     self.x1 = float(event.xdata)
  69 |                     self.y1 = float(event.ydata)
  70 |                     self.fig.canvas.draw_idle()
  71 |                     self.currently_selecting = False
  72 |                     self.rect.set_visible(False)
  73 |                     self.axes.patches[0].remove()
  74 |                     x_min = min(self.x0, self.x1)
  75 |                     x_max = max(self.x0, self.x1)
  76 |                     y_min = min(self.y0, self.y1)
  77 |                     y_max = max(self.y0, self.y1)
  78 |                     bbox = np.array([x_min, y_min, x_max, y_max])
  79 |                     with torch.no_grad():
  80 |                         seg = self._infer(bbox)
  81 |                         torch.cuda.empty_cache()
  82 |                     show_mask(seg, self.axes, random_color=random_color, alpha=alpha)
  83 |                     self.segs.append(deepcopy(seg))
  84 |                     del seg
  85 |                     self.rect = None
  86 |                     gc.collect()
  87 | 
  88 |         def __on_motion(event):
  89 |             if event.inaxes == self.axes:
  90 |                 if self.currently_selecting:
  91 |                     self.x1 = float(event.xdata)
  92 |                     self.y1 = float(event.ydata)
  93 |                     #add rectangle for selection here
  94 |                     self.rect.set_visible(True)
  95 |                     xlim = np.sort([self.x0, self.x1])
  96 |                     ylim = np.sort([self.y0, self.y1])
  97 |                     self.rect.set_xy((xlim[0],ylim[0] ) )
  98 |                     rect_width = np.diff(xlim)[0]
  99 |                     self.rect.set_width(rect_width)
 100 |                     rect_height = np.diff(ylim)[0]
 101 |                     self.rect.set_height(rect_height)
 102 | 
 103 |         clear_button = widgets.Button(description="clear")
 104 |         def __on_clear_button_clicked(b):
 105 |             for i in range(len(self.axes.images)):
 106 |                 self.axes.images[0].remove()
 107 |             self.axes.clear()
 108 |             self.axes.axis('off')
 109 |             self.axes.imshow(self.image)
 110 |             if len(self.axes.patches) > 0:
 111 |                 self.axes.patches[0].remove()
 112 |             self.segs = []
 113 |             self.fig.canvas.draw_idle()
 114 | 
 115 |         save_button = widgets.Button(description="save")
 116 |         def __on_save_button_clicked(b):
 117 |             plt.savefig("seg_result.png", bbox_inches='tight', pad_inches=0)
 118 |             if len(self.segs) > 0:
 119 |                 save_seg = np.zeros_like(self.segs[0])
 120 |                 for i, seg in enumerate(self.segs, start=1):
 121 |                     save_seg[seg > 0] = i
 122 |                 cv2.imwrite("segs.png", save_seg)
 123 |                 print(f"Segmentation result saved to {getcwd()}")
 124 |         
 125 |         display(clear_button)
 126 |         clear_button.on_click(__on_clear_button_clicked)
 127 | 
 128 |         self.fig.canvas.mpl_connect('button_press_event', __on_press)
 129 |         self.fig.canvas.mpl_connect('motion_notify_event', __on_motion)
 130 |         self.fig.canvas.mpl_connect('button_release_event', __on_release)
 131 | 
 132 |         plt.show()
 133 | 
 134 |         display(save_button)
 135 |         save_button.on_click(__on_save_button_clicked)
 136 | 
 137 |     def show(self, image_path, fig_size=5, random_color=True, alpha=0.65):
 138 |         self.set_image_path(image_path)
 139 |         self._show(fig_size=fig_size, random_color=random_color, alpha=alpha)
 140 | 
 141 |     def set_image_path(self, image_path):
 142 |         image = cv2.imread(image_path)
 143 |         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 144 |         self._set_image(image)
 145 |     
 146 |     def _set_image(self, image):
 147 |         self.image = image
 148 |         self.img_size = image.shape[:2]
 149 |         image_preprocess = self._preprocess_image(image)
 150 |         with torch.no_grad():
 151 |             self.image_embeddings = self.model.image_encoder(image_preprocess)
 152 | 
 153 |     def _preprocess_image(self, image):
 154 |         img_resize = cv2.resize(
 155 |             image,
 156 |             (1024, 1024),
 157 |             interpolation=cv2.INTER_CUBIC
 158 |         )
 159 |         # Resizing
 160 |         img_resize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8, a_max=None) # normalize to [0, 1], (H, W, 3
 161 |         # convert the shape to (3, H, W)
 162 |         assert np.max(img_resize)<=1.0 and np.min(img_resize)>=0.0, 'image should be normalized to [0, 1]'
 163 |         img_tensor = torch.tensor(img_resize).float().permute(2, 0, 1).unsqueeze(0).to(self.model.device)
 164 | 
 165 |         return img_tensor
 166 |     
 167 |     @torch.no_grad()
 168 |     def _infer(self, bbox):
 169 |         ori_H, ori_W = self.img_size
 170 |         scale_to_1024 = 1024 / np.array([ori_W, ori_H, ori_W, ori_H])
 171 |         bbox_1024 = bbox * scale_to_1024
 172 |         bbox_torch = torch.as_tensor(bbox_1024, dtype=torch.float).unsqueeze(0).to(self.model.device)
 173 |         if len(bbox_torch.shape) == 2:
 174 |             bbox_torch = bbox_torch.unsqueeze(1)
 175 |     
 176 |         sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
 177 |             points=None,
 178 |             boxes=bbox_torch,
 179 |             masks=None,
 180 |         )
 181 |         low_res_logits, _ = self.model.mask_decoder(
 182 |             image_embeddings = self.image_embeddings, # (B, 256, 64, 64)
 183 |             image_pe = self.model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
 184 |             sparse_prompt_embeddings = sparse_embeddings, # (B, 2, 256)
 185 |             dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
 186 |             multimask_output=False,
 187 |             )
 188 | 
 189 |         low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
 190 | 
 191 |         low_res_pred = F.interpolate(
 192 |             low_res_pred,
 193 |             size=self.img_size,
 194 |             mode="bilinear",
 195 |             align_corners=False,
 196 |         )  # (1, 1, gt.shape)
 197 |         low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
 198 |         medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
 199 |         return medsam_seg
 200 | 
 201 | 
 202 | 
 203 | class PointPromptDemo:
 204 |     def __init__(self, model, dataroot):
 205 |         self.model = model
 206 |         self.model.eval()
 207 |         self.image = None
 208 |         self.image_embeddings = None
 209 |         self.img_size = None
 210 |         self.img_name = None
 211 |         self.gt = None
 212 |         
 213 |         ## load demo data
 214 |         self.dataroot = dataroot
 215 |         self.img_path = join(dataroot, 'imgs')
 216 |         self.gt_path = join(dataroot, 'gts_ts')
 217 |         self.gt_path_files = sorted(glob(join(self.gt_path, '**/*.npy'), recursive=True))
 218 |         self.gt_path_files = [file for file in self.gt_path_files if isfile(join(self.img_path, basename(file)))]
 219 | 
 220 |     def show_mask(self, mask, ax, random_color=False, alpha=0.95):
 221 |         if random_color:
 222 |             color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
 223 |         else:
 224 |             color = np.array([251/255, 252/255, 30/255, alpha])
 225 |         h, w = mask.shape[-2:]
 226 |         mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
 227 |         ax.imshow(mask_image)
 228 | 
 229 |     @torch.no_grad()
 230 |     def infer(self, x, y):
 231 |         coords_1024 = np.array([[[
 232 |             x * 1024 / self.img_size[1],
 233 |             y * 1024 / self.img_size[0]
 234 |         ]]])
 235 |         coords_torch = torch.tensor(coords_1024, dtype=torch.float32).to(self.model.device)
 236 |         labels_torch = torch.tensor([[1]], dtype=torch.long).to(self.model.device)
 237 |         point_prompt = (coords_torch, labels_torch)
 238 | 
 239 |         sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
 240 |             points = point_prompt,
 241 |             boxes = None,
 242 |             masks = None,
 243 |         )
 244 |         low_res_logits, _ = self.model.mask_decoder(
 245 |             image_embeddings=self.image_embeddings, # (B, 256, 64, 64)
 246 |             image_pe=self.model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
 247 |             sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
 248 |             dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
 249 |             multimask_output=False,
 250 |         )
 251 | 
 252 |         low_res_probs = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
 253 |         low_res_pred = F.interpolate(
 254 |             low_res_probs,
 255 |             size = self.img_size,
 256 |             mode = 'bilinear',
 257 |             align_corners = False
 258 |         )
 259 |         low_res_pred = low_res_pred.detach().cpu().numpy().squeeze()
 260 | 
 261 |         seg = np.uint8(low_res_pred > 0.5)
 262 | 
 263 |         return seg
 264 | 
 265 |     def show(self, fig_size=3, alpha=0.95, scatter_size=25):
 266 | 
 267 |         assert self.image is not None, "Please set image first."
 268 |         fig, axes = plt.subplots(1, 2, figsize=(2 * fig_size, fig_size))
 269 |         fig.canvas.header_visible = False
 270 |         fig.canvas.footer_visible = False
 271 |         fig.canvas.toolbar_visible = False
 272 |         fig.canvas.resizable = False
 273 | 
 274 |         plt.tight_layout()
 275 | 
 276 |         for i in range(2):
 277 |             axes[i].imshow(self.image)
 278 |             axes[i].axis('off')
 279 |             if i == 0:
 280 |                 axes[i].set_title('Ground Truth')
 281 |             else:
 282 |                 axes[i].set_title('Segmentation')
 283 | 
 284 |         def onclick(event):
 285 |             if event.inaxes == axes[1]:
 286 |                 x, y = float(event.xdata), float(event.ydata)
 287 |                 with torch.no_grad():
 288 |                     ## rescale x, y from canvas size to 1024 x 1024
 289 |                     seg = self.infer(x, y)
 290 | 
 291 |                 for i in range(2):
 292 |                     axes[i].clear()
 293 |                     axes[i].imshow(self.image)
 294 |                     axes[i].axis('off')
 295 |                     axes[i].scatter(x, y, c='r', s=scatter_size)
 296 | 
 297 |                 axes[1].set_title('Segmentation')
 298 |                 self.show_mask(seg, axes[1], random_color=False, alpha=alpha)
 299 | 
 300 |                 axes[0].set_title('Ground Truth')
 301 |                 gt_label_id = self.get_label_id((x, y))
 302 |                 if gt_label_id > 0:
 303 |                     gt_show = np.uint8(self.gt == gt_label_id)
 304 |                     self.show_mask(gt_show, axes[0], random_color=False, alpha=alpha)
 305 | 
 306 |                 gc.collect()
 307 | 
 308 |         fig.canvas.mpl_connect('button_press_event', onclick)
 309 |         plt.show()
 310 | 
 311 |     def set_image(self, image_index):
 312 |         image_path = join(self.img_path, basename(self.gt_path_files[image_index]))
 313 |         image = np.load(image_path)
 314 |         self.image = image
 315 |         self.img_size = image.shape[:2]
 316 |         self.img_name = basename(image_path)
 317 |         image_preprocess = self.preprocess_image(image)
 318 |         with torch.no_grad():
 319 |             self.image_embeddings = self.model.image_encoder(image_preprocess)
 320 |         
 321 |         gt_path = self.gt_path_files[image_index]
 322 |         gt = np.load(gt_path)
 323 |         gt_resize = cv2.resize(
 324 |             gt,
 325 |             (self.img_size[1], self.img_size[0]),
 326 |             interpolation=cv2.INTER_NEAREST
 327 |         )
 328 |         self.gt = gt_resize
 329 | 
 330 |     def preprocess_image(self, image):
 331 |         img_resize = cv2.resize(
 332 |             image,
 333 |             (1024, 1024),
 334 |             interpolation=cv2.INTER_CUBIC
 335 |         )
 336 |         # Resizing
 337 |         img_resize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8, a_max=None) # normalize to [0, 1], (H, W, 3
 338 |         # convert the shape to (3, H, W)
 339 |         assert np.max(img_resize)<=1.0 and np.min(img_resize)>=0.0, 'image should be normalized to [0, 1]'
 340 |         img_tensor = torch.tensor(img_resize).float().permute(2, 0, 1).unsqueeze(0).to(self.model.device)
 341 | 
 342 |         return img_tensor
 343 |     
 344 |     def get_label_id(self, coords):
 345 |         x, y = coords
 346 |         label_id = self.gt[int(y), int(x)]
 347 | 
 348 |         return label_id
 349 | 
 350 | 
 351 | class TextPromptDemo:
 352 |     def __init__(self, model, dataroot):
 353 |         ## Requires transformers only if users would like to try text prompt
 354 |         from transformers import CLIPTokenizer
 355 |         self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")   
 356 |         self.model = model
 357 |         self.model.eval()
 358 |         self.image = None
 359 |         self.image_embeddings = None
 360 |         self.img_size = None
 361 |         self.img_name = None
 362 |         self.gt = None
 363 |         self.label_dict = {
 364 |             1: ["Liver", "liver"],
 365 |             2: ["Right Kidney", "right kidney"],
 366 |             3: ["Spleen", "spleen"],
 367 |             4: ["Pancreas", "pancreas"],
 368 |             5: ["Aorta", "aorta"],
 369 |             6: ["Inferior Vena Cava", "IVC", "inferior vena cava", "ivc"],
 370 |             7: ["Right Adrenal Gland", "RAG", "right adrenal gland", "rag"],
 371 |             8: ["Left Adrenal Gland", "LAG", "left adrenal gland", "lag"],
 372 |             9: ["Gallbladder", "gallbladder"],
 373 |             10: ["Esophagus", "esophagus"],
 374 |             11: ["Stomach", "stomach"],
 375 |             12: ["Duodenum", "duodenum"],
 376 |             13: ["Left Kidney", "left kidney"]
 377 |         }
 378 |         self.caption_label_dict = {}
 379 |         for label_id, label_list in self.label_dict.items():
 380 |             for label in label_list:
 381 |                 self.caption_label_dict[label] = label_id
 382 |         
 383 |         ## load demo data
 384 |         self.dataroot = dataroot
 385 |         self.img_path = join(dataroot, 'imgs')
 386 |         self.gt_path = join(dataroot, 'gts_ts')
 387 |         self.gt_path_files = sorted(glob(join(self.gt_path, '**/*.npy'), recursive=True))
 388 |         self.gt_path_files = [file for file in self.gt_path_files if isfile(join(self.img_path, basename(file)))]
 389 | 
 390 |     def show_mask(self, mask, ax, random_color=False, alpha=0.95):
 391 |         if random_color:
 392 |             color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
 393 |         else:
 394 |             color = np.array([251/255, 252/255, 30/255, alpha])
 395 |         h, w = mask.shape[-2:]
 396 |         mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
 397 |         ax.imshow(mask_image)
 398 |     
 399 |     @torch.no_grad()
 400 |     def infer(self, text):
 401 |         tokens = self.tokenize_text(text).to(self.model.device)
 402 |         sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
 403 |             points = None,
 404 |             boxes = None,
 405 |             masks = None,
 406 |             tokens = tokens
 407 |         )
 408 |         low_res_logits, _ = self.model.mask_decoder(
 409 |             image_embeddings=self.image_embeddings, # (B, 256, 64, 64)
 410 |             image_pe=self.model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
 411 |             sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
 412 |             dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
 413 |             multimask_output=False,
 414 |             )
 415 |         low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
 416 |         low_res_pred = F.interpolate(
 417 |             low_res_pred,
 418 |             size = self.img_size,
 419 |             mode = 'bilinear',
 420 |             align_corners = False
 421 |         )
 422 |         low_res_pred = low_res_pred.detach().cpu().numpy().squeeze()
 423 |         seg = np.uint8(low_res_pred > 0.5)
 424 | 
 425 |         return seg
 426 | 
 427 |     def show(self, fig_size=3, alpha=0.95):
 428 |         assert self.image is not None, "Please set image first."
 429 |         fig, axes = plt.subplots(1, 2, figsize=(2 * fig_size, fig_size))
 430 |         fig.canvas.header_visible = False
 431 |         fig.canvas.footer_visible = False
 432 |         fig.canvas.toolbar_visible = False
 433 |         fig.canvas.resizable = False
 434 | 
 435 |         avil_ids = np.unique(self.gt)[1:]
 436 |         avail_prompts = []
 437 |         for id in avil_ids:
 438 |             avail_prompts += self.label_dict[id]
 439 |         print("Possible prompts: ", avail_prompts)
 440 | 
 441 |         plt.tight_layout()
 442 | 
 443 |         for i in range(2):
 444 |             axes[i].imshow(self.image)
 445 |             axes[i].axis('off')
 446 |             if i == 0:
 447 |                 axes[i].set_title('Ground Truth')
 448 |             else:
 449 |                 axes[i].set_title('Segmentation')
 450 | 
 451 |         text = widgets.Text(
 452 |             value='',
 453 |             placeholder='Prompt',
 454 |             description='Prompt',
 455 |             disabled=False
 456 |         )
 457 |         display(text)
 458 | 
 459 |         def callback(wget):
 460 |             for i in range(2):
 461 |                 axes[i].clear()
 462 |                 axes[i].imshow(self.image)
 463 |                 axes[i].axis('off')
 464 |             caption = wget.value
 465 |             seg = self.infer(caption)
 466 |             axes[1].set_title('Segmentation')
 467 |             self.show_mask(seg, axes[1], random_color=False, alpha=alpha)
 468 | 
 469 |             axes[0].set_title('Ground Truth')
 470 |             try:
 471 |                 gt_label_id = self.caption_label_dict[caption]
 472 |             except:
 473 |                 gt_label_id = self.guess_gt_label_id(self.gt, seg)
 474 |             if gt_label_id in self.label_dict:
 475 |                 gt_show = np.uint8(self.gt == gt_label_id)
 476 |                 self.show_mask(gt_show, axes[0], random_color=False, alpha=alpha)
 477 |             else:
 478 |                 axes[0].clear()
 479 |                 axes[0].imshow(self.image)
 480 |                 axes[0].axis('off')
 481 |                 axes[0].set_title('Ground Truth')
 482 | 
 483 |         text.on_submit(callback)
 484 | 
 485 |     def guess_gt_label_id(self, gt, seg):
 486 |         mask_area = seg > 0
 487 |         gt_area = gt[mask_area]
 488 |         gt_label_id = np.argmax(np.bincount(gt_area))
 489 | 
 490 |         return gt_label_id
 491 | 
 492 |     def set_image(self, image_index):
 493 |         image_path = join(self.img_path, basename(self.gt_path_files[image_index]))
 494 |         image = np.load(image_path)
 495 |         self.image = image
 496 |         self.img_size = image.shape[:2]
 497 |         self.img_name = basename(image_path)
 498 |         image_preprocess = self.preprocess_image(image)
 499 |         with torch.no_grad():
 500 |             self.image_embeddings = self.model.image_encoder(image_preprocess)
 501 |         
 502 |         gt_path = self.gt_path_files[image_index]
 503 |         gt = np.load(gt_path)
 504 |         gt_resize = cv2.resize(
 505 |             gt,
 506 |             (self.img_size[1], self.img_size[0]),
 507 |             interpolation=cv2.INTER_NEAREST
 508 |         )
 509 |         self.gt = gt_resize
 510 | 
 511 |     def preprocess_image(self, image):
 512 |         img_resize = cv2.resize(
 513 |             image,
 514 |             (1024, 1024),
 515 |             interpolation=cv2.INTER_CUBIC
 516 |         )
 517 |         # Resizing
 518 |         img_resize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8, a_max=None) # normalize to [0, 1], (H, W, 3
 519 |         # convert the shape to (3, H, W)
 520 |         assert np.max(img_resize)<=1.0 and np.min(img_resize)>=0.0, 'image should be normalized to [0, 1]'
 521 |         img_tensor = torch.tensor(img_resize).float().permute(2, 0, 1).unsqueeze(0).to(self.model.device)
 522 | 
 523 |         return img_tensor
 524 |     
 525 | 
 526 |     def tokenize_text(self, text):
 527 |         """
 528 |         Tokenize text using CLIP tokenizer
 529 |         """
 530 |         return self.tokenizer(
 531 |             text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt" 
 532 |         ).input_ids.squeeze(1)
```
