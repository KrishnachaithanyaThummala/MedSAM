# segment_anything/predictor.py

```python
   1 | # -*- coding: utf-8 -*-
   2 | # Copyright (c) Meta Platforms, Inc. and affiliates.
   3 | # All rights reserved.
   4 | 
   5 | # This source code is licensed under the license found in the
   6 | # LICENSE file in the root directory of this source tree.
   7 | 
   8 | import numpy as np
   9 | import torch
  10 | 
  11 | from segment_anything.modeling import Sam
  12 | 
  13 | from typing import Optional, Tuple
  14 | 
  15 | from .utils.transforms import ResizeLongestSide
  16 | 
  17 | 
  18 | class SamPredictor:
  19 |     def __init__(
  20 |         self,
  21 |         sam_model: Sam,
  22 |     ) -> None:
  23 |         """
  24 |         Uses SAM to calculate the image embedding for an image, and then
  25 |         allow repeated, efficient mask prediction given prompts.
  26 | 
  27 |         Arguments:
  28 |           sam_model (Sam): The model to use for mask prediction.
  29 |         """
  30 |         super().__init__()
  31 |         self.model = sam_model
  32 |         self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
  33 |         self.reset_image()
  34 | 
  35 |     def set_image(
  36 |         self,
  37 |         image: np.ndarray,
  38 |         image_format: str = "RGB",
  39 |     ) -> None:
  40 |         """
  41 |         Calculates the image embeddings for the provided image, allowing
  42 |         masks to be predicted with the 'predict' method.
  43 | 
  44 |         Arguments:
  45 |           image (np.ndarray): The image for calculating masks. Expects an
  46 |             image in HWC uint8 format, with pixel values in [0, 255].
  47 |           image_format (str): The color format of the image, in ['RGB', 'BGR'].
  48 |         """
  49 |         assert image_format in [
  50 |             "RGB",
  51 |             "BGR",
  52 |         ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
  53 |         if image_format != self.model.image_format:
  54 |             image = image[..., ::-1]
  55 | 
  56 |         # Transform the image to the form expected by the model
  57 |         input_image = self.transform.apply_image(image)
  58 |         input_image_torch = torch.as_tensor(input_image, device=self.device)
  59 |         input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[
  60 |             None, :, :, :
  61 |         ]
  62 | 
  63 |         self.set_torch_image(input_image_torch, image.shape[:2])
  64 | 
  65 |     @torch.no_grad()
  66 |     def set_torch_image(
  67 |         self,
  68 |         transformed_image: torch.Tensor,
  69 |         original_image_size: Tuple[int, ...],
  70 |     ) -> None:
  71 |         """
  72 |         Calculates the image embeddings for the provided image, allowing
  73 |         masks to be predicted with the 'predict' method. Expects the input
  74 |         image to be already transformed to the format expected by the model.
  75 | 
  76 |         Arguments:
  77 |           transformed_image (torch.Tensor): The input image, with shape
  78 |             1x3xHxW, which has been transformed with ResizeLongestSide.
  79 |           original_image_size (tuple(int, int)): The size of the image
  80 |             before transformation, in (H, W) format.
  81 |         """
  82 |         assert (
  83 |             len(transformed_image.shape) == 4
  84 |             and transformed_image.shape[1] == 3
  85 |             and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
  86 |         ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."
  87 |         self.reset_image()
  88 | 
  89 |         self.original_size = original_image_size
  90 |         self.input_size = tuple(transformed_image.shape[-2:])
  91 |         input_image = self.model.preprocess(transformed_image)
  92 |         self.features = self.model.image_encoder(input_image)
  93 |         self.is_image_set = True
  94 | 
  95 |     def predict(
  96 |         self,
  97 |         point_coords: Optional[np.ndarray] = None,
  98 |         point_labels: Optional[np.ndarray] = None,
  99 |         box: Optional[np.ndarray] = None,
 100 |         mask_input: Optional[np.ndarray] = None,
 101 |         multimask_output: bool = True,
 102 |         return_logits: bool = False,
 103 |     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
 104 |         """
 105 |         Predict masks for the given input prompts, using the currently set image.
 106 | 
 107 |         Arguments:
 108 |           point_coords (np.ndarray or None): A Nx2 array of point prompts to the
 109 |             model. Each point is in (X,Y) in pixels.
 110 |           point_labels (np.ndarray or None): A length N array of labels for the
 111 |             point prompts. 1 indicates a foreground point and 0 indicates a
 112 |             background point.
 113 |           box (np.ndarray or None): A length 4 array given a box prompt to the
 114 |             model, in XYXY format.
 115 |           mask_input (np.ndarray): A low resolution mask input to the model, typically
 116 |             coming from a previous prediction iteration. Has form 1xHxW, where
 117 |             for SAM, H=W=256.
 118 |           multimask_output (bool): If true, the model will return three masks.
 119 |             For ambiguous input prompts (such as a single click), this will often
 120 |             produce better masks than a single prediction. If only a single
 121 |             mask is needed, the model's predicted quality score can be used
 122 |             to select the best mask. For non-ambiguous prompts, such as multiple
 123 |             input prompts, multimask_output=False can give better results.
 124 |           return_logits (bool): If true, returns un-thresholded masks logits
 125 |             instead of a binary mask.
 126 | 
 127 |         Returns:
 128 |           (np.ndarray): The output masks in CxHxW format, where C is the
 129 |             number of masks, and (H, W) is the original image size.
 130 |           (np.ndarray): An array of length C containing the model's
 131 |             predictions for the quality of each mask.
 132 |           (np.ndarray): An array of shape CxHxW, where C is the number
 133 |             of masks and H=W=256. These low resolution logits can be passed to
 134 |             a subsequent iteration as mask input.
 135 |         """
 136 |         if not self.is_image_set:
 137 |             raise RuntimeError(
 138 |                 "An image must be set with .set_image(...) before mask prediction."
 139 |             )
 140 | 
 141 |         # Transform input prompts
 142 |         coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
 143 |         if point_coords is not None:
 144 |             assert (
 145 |                 point_labels is not None
 146 |             ), "point_labels must be supplied if point_coords is supplied."
 147 |             point_coords = self.transform.apply_coords(point_coords, self.original_size)
 148 |             coords_torch = torch.as_tensor(
 149 |                 point_coords, dtype=torch.float, device=self.device
 150 |             )
 151 |             labels_torch = torch.as_tensor(
 152 |                 point_labels, dtype=torch.int, device=self.device
 153 |             )
 154 |             coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
 155 |         if box is not None:
 156 |             box = self.transform.apply_boxes(box, self.original_size)
 157 |             box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
 158 |             box_torch = box_torch[None, :]
 159 |         if mask_input is not None:
 160 |             mask_input_torch = torch.as_tensor(
 161 |                 mask_input, dtype=torch.float, device=self.device
 162 |             )
 163 |             mask_input_torch = mask_input_torch[None, :, :, :]
 164 | 
 165 |         masks, iou_predictions, low_res_masks = self.predict_torch(
 166 |             coords_torch,
 167 |             labels_torch,
 168 |             box_torch,
 169 |             mask_input_torch,
 170 |             multimask_output,
 171 |             return_logits=return_logits,
 172 |         )
 173 | 
 174 |         masks_np = masks[0].detach().cpu().numpy()
 175 |         iou_predictions_np = iou_predictions[0].detach().cpu().numpy()
 176 |         low_res_masks_np = low_res_masks[0].detach().cpu().numpy()
 177 |         return masks_np, iou_predictions_np, low_res_masks_np
 178 | 
 179 |     @torch.no_grad()
 180 |     def predict_torch(
 181 |         self,
 182 |         point_coords: Optional[torch.Tensor],
 183 |         point_labels: Optional[torch.Tensor],
 184 |         boxes: Optional[torch.Tensor] = None,
 185 |         mask_input: Optional[torch.Tensor] = None,
 186 |         multimask_output: bool = True,
 187 |         return_logits: bool = False,
 188 |     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
 189 |         """
 190 |         Predict masks for the given input prompts, using the currently set image.
 191 |         Input prompts are batched torch tensors and are expected to already be
 192 |         transformed to the input frame using ResizeLongestSide.
 193 | 
 194 |         Arguments:
 195 |           point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
 196 |             model. Each point is in (X,Y) in pixels.
 197 |           point_labels (torch.Tensor or None): A BxN array of labels for the
 198 |             point prompts. 1 indicates a foreground point and 0 indicates a
 199 |             background point.
 200 |           boxes (np.ndarray or None): A Bx4 array given a box prompt to the
 201 |             model, in XYXY format.
 202 |           mask_input (np.ndarray): A low resolution mask input to the model, typically
 203 |             coming from a previous prediction iteration. Has form Bx1xHxW, where
 204 |             for SAM, H=W=256. Masks returned by a previous iteration of the
 205 |             predict method do not need further transformation.
 206 |           multimask_output (bool): If true, the model will return three masks.
 207 |             For ambiguous input prompts (such as a single click), this will often
 208 |             produce better masks than a single prediction. If only a single
 209 |             mask is needed, the model's predicted quality score can be used
 210 |             to select the best mask. For non-ambiguous prompts, such as multiple
 211 |             input prompts, multimask_output=False can give better results.
 212 |           return_logits (bool): If true, returns un-thresholded masks logits
 213 |             instead of a binary mask.
 214 | 
 215 |         Returns:
 216 |           (torch.Tensor): The output masks in BxCxHxW format, where C is the
 217 |             number of masks, and (H, W) is the original image size.
 218 |           (torch.Tensor): An array of shape BxC containing the model's
 219 |             predictions for the quality of each mask.
 220 |           (torch.Tensor): An array of shape BxCxHxW, where C is the number
 221 |             of masks and H=W=256. These low res logits can be passed to
 222 |             a subsequent iteration as mask input.
 223 |         """
 224 |         if not self.is_image_set:
 225 |             raise RuntimeError(
 226 |                 "An image must be set with .set_image(...) before mask prediction."
 227 |             )
 228 | 
 229 |         if point_coords is not None:
 230 |             points = (point_coords, point_labels)
 231 |         else:
 232 |             points = None
 233 | 
 234 |         # Embed prompts
 235 |         sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
 236 |             points=points,
 237 |             boxes=boxes,
 238 |             masks=mask_input,
 239 |         )
 240 | 
 241 |         # Predict masks
 242 |         low_res_masks, iou_predictions = self.model.mask_decoder(
 243 |             image_embeddings=self.features,
 244 |             image_pe=self.model.prompt_encoder.get_dense_pe(),
 245 |             sparse_prompt_embeddings=sparse_embeddings,
 246 |             dense_prompt_embeddings=dense_embeddings,
 247 |             multimask_output=multimask_output,
 248 |         )
 249 | 
 250 |         # Upscale the masks to the original image resolution
 251 |         masks = self.model.postprocess_masks(
 252 |             low_res_masks, self.input_size, self.original_size
 253 |         )
 254 | 
 255 |         if not return_logits:
 256 |             masks = masks > self.model.mask_threshold
 257 | 
 258 |         return masks, iou_predictions, low_res_masks
 259 | 
 260 |     def get_image_embedding(self) -> torch.Tensor:
 261 |         """
 262 |         Returns the image embeddings for the currently set image, with
 263 |         shape 1xCxHxW, where C is the embedding dimension and (H,W) are
 264 |         the embedding spatial dimension of SAM (typically C=256, H=W=64).
 265 |         """
 266 |         if not self.is_image_set:
 267 |             raise RuntimeError(
 268 |                 "An image must be set with .set_image(...) to generate an embedding."
 269 |             )
 270 |         assert (
 271 |             self.features is not None
 272 |         ), "Features must exist if an image has been set."
 273 |         return self.features
 274 | 
 275 |     @property
 276 |     def device(self) -> torch.device:
 277 |         return self.model.device
 278 | 
 279 |     def reset_image(self) -> None:
 280 |         """Resets the currently set image."""
 281 |         self.is_image_set = False
 282 |         self.features = None
 283 |         self.orig_h = None
 284 |         self.orig_w = None
 285 |         self.input_h = None
 286 |         self.input_w = None
```
