# segment_anything/automatic_mask_generator.py

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
  10 | from torchvision.ops.boxes import batched_nms, box_area  # type: ignore
  11 | 
  12 | from typing import Any, Dict, List, Optional, Tuple
  13 | 
  14 | from .modeling import Sam
  15 | from .predictor import SamPredictor
  16 | from .utils.amg import (
  17 |     MaskData,
  18 |     area_from_rle,
  19 |     batch_iterator,
  20 |     batched_mask_to_box,
  21 |     box_xyxy_to_xywh,
  22 |     build_all_layer_point_grids,
  23 |     calculate_stability_score,
  24 |     coco_encode_rle,
  25 |     generate_crop_boxes,
  26 |     is_box_near_crop_edge,
  27 |     mask_to_rle_pytorch,
  28 |     remove_small_regions,
  29 |     rle_to_mask,
  30 |     uncrop_boxes_xyxy,
  31 |     uncrop_masks,
  32 |     uncrop_points,
  33 | )
  34 | 
  35 | 
  36 | class SamAutomaticMaskGenerator:
  37 |     def __init__(
  38 |         self,
  39 |         model: Sam,
  40 |         points_per_side: Optional[int] = 32,
  41 |         points_per_batch: int = 64,
  42 |         pred_iou_thresh: float = 0.88,
  43 |         stability_score_thresh: float = 0.95,
  44 |         stability_score_offset: float = 1.0,
  45 |         box_nms_thresh: float = 0.7,
  46 |         crop_n_layers: int = 0,
  47 |         crop_nms_thresh: float = 0.7,
  48 |         crop_overlap_ratio: float = 512 / 1500,
  49 |         crop_n_points_downscale_factor: int = 1,
  50 |         point_grids: Optional[List[np.ndarray]] = None,
  51 |         min_mask_region_area: int = 0,
  52 |         output_mode: str = "binary_mask",
  53 |     ) -> None:
  54 |         """
  55 |         Using a SAM model, generates masks for the entire image.
  56 |         Generates a grid of point prompts over the image, then filters
  57 |         low quality and duplicate masks. The default settings are chosen
  58 |         for SAM with a ViT-H backbone.
  59 | 
  60 |         Arguments:
  61 |           model (Sam): The SAM model to use for mask prediction.
  62 |           points_per_side (int or None): The number of points to be sampled
  63 |             along one side of the image. The total number of points is
  64 |             points_per_side**2. If None, 'point_grids' must provide explicit
  65 |             point sampling.
  66 |           points_per_batch (int): Sets the number of points run simultaneously
  67 |             by the model. Higher numbers may be faster but use more GPU memory.
  68 |           pred_iou_thresh (float): A filtering threshold in [0,1], using the
  69 |             model's predicted mask quality.
  70 |           stability_score_thresh (float): A filtering threshold in [0,1], using
  71 |             the stability of the mask under changes to the cutoff used to binarize
  72 |             the model's mask predictions.
  73 |           stability_score_offset (float): The amount to shift the cutoff when
  74 |             calculated the stability score.
  75 |           box_nms_thresh (float): The box IoU cutoff used by non-maximal
  76 |             suppression to filter duplicate masks.
  77 |           crop_n_layers (int): If >0, mask prediction will be run again on
  78 |             crops of the image. Sets the number of layers to run, where each
  79 |             layer has 2**i_layer number of image crops.
  80 |           crop_nms_thresh (float): The box IoU cutoff used by non-maximal
  81 |             suppression to filter duplicate masks between different crops.
  82 |           crop_overlap_ratio (float): Sets the degree to which crops overlap.
  83 |             In the first crop layer, crops will overlap by this fraction of
  84 |             the image length. Later layers with more crops scale down this overlap.
  85 |           crop_n_points_downscale_factor (int): The number of points-per-side
  86 |             sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
  87 |           point_grids (list(np.ndarray) or None): A list over explicit grids
  88 |             of points used for sampling, normalized to [0,1]. The nth grid in the
  89 |             list is used in the nth crop layer. Exclusive with points_per_side.
  90 |           min_mask_region_area (int): If >0, postprocessing will be applied
  91 |             to remove disconnected regions and holes in masks with area smaller
  92 |             than min_mask_region_area. Requires opencv.
  93 |           output_mode (str): The form masks are returned in. Can be 'binary_mask',
  94 |             'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
  95 |             For large resolutions, 'binary_mask' may consume large amounts of
  96 |             memory.
  97 |         """
  98 | 
  99 |         assert (points_per_side is None) != (
 100 |             point_grids is None
 101 |         ), "Exactly one of points_per_side or point_grid must be provided."
 102 |         if points_per_side is not None:
 103 |             self.point_grids = build_all_layer_point_grids(
 104 |                 points_per_side,
 105 |                 crop_n_layers,
 106 |                 crop_n_points_downscale_factor,
 107 |             )
 108 |         elif point_grids is not None:
 109 |             self.point_grids = point_grids
 110 |         else:
 111 |             raise ValueError("Can't have both points_per_side and point_grid be None.")
 112 | 
 113 |         assert output_mode in [
 114 |             "binary_mask",
 115 |             "uncompressed_rle",
 116 |             "coco_rle",
 117 |         ], f"Unknown output_mode {output_mode}."
 118 |         if output_mode == "coco_rle":
 119 |             from pycocotools import mask as mask_utils  # type: ignore # noqa: F401
 120 | 
 121 |         if min_mask_region_area > 0:
 122 |             import cv2  # type: ignore # noqa: F401
 123 | 
 124 |         self.predictor = SamPredictor(model)
 125 |         self.points_per_batch = points_per_batch
 126 |         self.pred_iou_thresh = pred_iou_thresh
 127 |         self.stability_score_thresh = stability_score_thresh
 128 |         self.stability_score_offset = stability_score_offset
 129 |         self.box_nms_thresh = box_nms_thresh
 130 |         self.crop_n_layers = crop_n_layers
 131 |         self.crop_nms_thresh = crop_nms_thresh
 132 |         self.crop_overlap_ratio = crop_overlap_ratio
 133 |         self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
 134 |         self.min_mask_region_area = min_mask_region_area
 135 |         self.output_mode = output_mode
 136 | 
 137 |     @torch.no_grad()
 138 |     def generate(self, image: np.ndarray) -> List[Dict[str, Any]]:
 139 |         """
 140 |         Generates masks for the given image.
 141 | 
 142 |         Arguments:
 143 |           image (np.ndarray): The image to generate masks for, in HWC uint8 format.
 144 | 
 145 |         Returns:
 146 |            list(dict(str, any)): A list over records for masks. Each record is
 147 |              a dict containing the following keys:
 148 |                segmentation (dict(str, any) or np.ndarray): The mask. If
 149 |                  output_mode='binary_mask', is an array of shape HW. Otherwise,
 150 |                  is a dictionary containing the RLE.
 151 |                bbox (list(float)): The box around the mask, in XYWH format.
 152 |                area (int): The area in pixels of the mask.
 153 |                predicted_iou (float): The model's own prediction of the mask's
 154 |                  quality. This is filtered by the pred_iou_thresh parameter.
 155 |                point_coords (list(list(float))): The point coordinates input
 156 |                  to the model to generate this mask.
 157 |                stability_score (float): A measure of the mask's quality. This
 158 |                  is filtered on using the stability_score_thresh parameter.
 159 |                crop_box (list(float)): The crop of the image used to generate
 160 |                  the mask, given in XYWH format.
 161 |         """
 162 | 
 163 |         # Generate masks
 164 |         mask_data = self._generate_masks(image)
 165 | 
 166 |         # Filter small disconnected regions and holes in masks
 167 |         if self.min_mask_region_area > 0:
 168 |             mask_data = self.postprocess_small_regions(
 169 |                 mask_data,
 170 |                 self.min_mask_region_area,
 171 |                 max(self.box_nms_thresh, self.crop_nms_thresh),
 172 |             )
 173 | 
 174 |         # Encode masks
 175 |         if self.output_mode == "coco_rle":
 176 |             mask_data["segmentations"] = [
 177 |                 coco_encode_rle(rle) for rle in mask_data["rles"]
 178 |             ]
 179 |         elif self.output_mode == "binary_mask":
 180 |             mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
 181 |         else:
 182 |             mask_data["segmentations"] = mask_data["rles"]
 183 | 
 184 |         # Write mask records
 185 |         curr_anns = []
 186 |         for idx in range(len(mask_data["segmentations"])):
 187 |             ann = {
 188 |                 "segmentation": mask_data["segmentations"][idx],
 189 |                 "area": area_from_rle(mask_data["rles"][idx]),
 190 |                 "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
 191 |                 "predicted_iou": mask_data["iou_preds"][idx].item(),
 192 |                 "point_coords": [mask_data["points"][idx].tolist()],
 193 |                 "stability_score": mask_data["stability_score"][idx].item(),
 194 |                 "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
 195 |             }
 196 |             curr_anns.append(ann)
 197 | 
 198 |         return curr_anns
 199 | 
 200 |     def _generate_masks(self, image: np.ndarray) -> MaskData:
 201 |         orig_size = image.shape[:2]
 202 |         crop_boxes, layer_idxs = generate_crop_boxes(
 203 |             orig_size, self.crop_n_layers, self.crop_overlap_ratio
 204 |         )
 205 | 
 206 |         # Iterate over image crops
 207 |         data = MaskData()
 208 |         for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
 209 |             crop_data = self._process_crop(image, crop_box, layer_idx, orig_size)
 210 |             data.cat(crop_data)
 211 | 
 212 |         # Remove duplicate masks between crops
 213 |         if len(crop_boxes) > 1:
 214 |             # Prefer masks from smaller crops
 215 |             scores = 1 / box_area(data["crop_boxes"])
 216 |             scores = scores.to(data["boxes"].device)
 217 |             keep_by_nms = batched_nms(
 218 |                 data["boxes"].float(),
 219 |                 scores,
 220 |                 torch.zeros_like(data["boxes"][:, 0]),  # categories
 221 |                 iou_threshold=self.crop_nms_thresh,
 222 |             )
 223 |             data.filter(keep_by_nms)
 224 | 
 225 |         data.to_numpy()
 226 |         return data
 227 | 
 228 |     def _process_crop(
 229 |         self,
 230 |         image: np.ndarray,
 231 |         crop_box: List[int],
 232 |         crop_layer_idx: int,
 233 |         orig_size: Tuple[int, ...],
 234 |     ) -> MaskData:
 235 |         # Crop the image and calculate embeddings
 236 |         x0, y0, x1, y1 = crop_box
 237 |         cropped_im = image[y0:y1, x0:x1, :]
 238 |         cropped_im_size = cropped_im.shape[:2]
 239 |         self.predictor.set_image(cropped_im)
 240 | 
 241 |         # Get points for this crop
 242 |         points_scale = np.array(cropped_im_size)[None, ::-1]
 243 |         points_for_image = self.point_grids[crop_layer_idx] * points_scale
 244 | 
 245 |         # Generate masks for this crop in batches
 246 |         data = MaskData()
 247 |         for (points,) in batch_iterator(self.points_per_batch, points_for_image):
 248 |             batch_data = self._process_batch(
 249 |                 points, cropped_im_size, crop_box, orig_size
 250 |             )
 251 |             data.cat(batch_data)
 252 |             del batch_data
 253 |         self.predictor.reset_image()
 254 | 
 255 |         # Remove duplicates within this crop.
 256 |         keep_by_nms = batched_nms(
 257 |             data["boxes"].float(),
 258 |             data["iou_preds"],
 259 |             torch.zeros_like(data["boxes"][:, 0]),  # categories
 260 |             iou_threshold=self.box_nms_thresh,
 261 |         )
 262 |         data.filter(keep_by_nms)
 263 | 
 264 |         # Return to the original image frame
 265 |         data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
 266 |         data["points"] = uncrop_points(data["points"], crop_box)
 267 |         data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])
 268 | 
 269 |         return data
 270 | 
 271 |     def _process_batch(
 272 |         self,
 273 |         points: np.ndarray,
 274 |         im_size: Tuple[int, ...],
 275 |         crop_box: List[int],
 276 |         orig_size: Tuple[int, ...],
 277 |     ) -> MaskData:
 278 |         orig_h, orig_w = orig_size
 279 | 
 280 |         # Run model on this batch
 281 |         transformed_points = self.predictor.transform.apply_coords(points, im_size)
 282 |         in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
 283 |         in_labels = torch.ones(
 284 |             in_points.shape[0], dtype=torch.int, device=in_points.device
 285 |         )
 286 |         masks, iou_preds, _ = self.predictor.predict_torch(
 287 |             in_points[:, None, :],
 288 |             in_labels[:, None],
 289 |             multimask_output=True,
 290 |             return_logits=True,
 291 |         )
 292 | 
 293 |         # Serialize predictions and store in MaskData
 294 |         data = MaskData(
 295 |             masks=masks.flatten(0, 1),
 296 |             iou_preds=iou_preds.flatten(0, 1),
 297 |             points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
 298 |         )
 299 |         del masks
 300 | 
 301 |         # Filter by predicted IoU
 302 |         if self.pred_iou_thresh > 0.0:
 303 |             keep_mask = data["iou_preds"] > self.pred_iou_thresh
 304 |             data.filter(keep_mask)
 305 | 
 306 |         # Calculate stability score
 307 |         data["stability_score"] = calculate_stability_score(
 308 |             data["masks"],
 309 |             self.predictor.model.mask_threshold,
 310 |             self.stability_score_offset,
 311 |         )
 312 |         if self.stability_score_thresh > 0.0:
 313 |             keep_mask = data["stability_score"] >= self.stability_score_thresh
 314 |             data.filter(keep_mask)
 315 | 
 316 |         # Threshold masks and calculate boxes
 317 |         data["masks"] = data["masks"] > self.predictor.model.mask_threshold
 318 |         data["boxes"] = batched_mask_to_box(data["masks"])
 319 | 
 320 |         # Filter boxes that touch crop boundaries
 321 |         keep_mask = ~is_box_near_crop_edge(
 322 |             data["boxes"], crop_box, [0, 0, orig_w, orig_h]
 323 |         )
 324 |         if not torch.all(keep_mask):
 325 |             data.filter(keep_mask)
 326 | 
 327 |         # Compress to RLE
 328 |         data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
 329 |         data["rles"] = mask_to_rle_pytorch(data["masks"])
 330 |         del data["masks"]
 331 | 
 332 |         return data
 333 | 
 334 |     @staticmethod
 335 |     def postprocess_small_regions(
 336 |         mask_data: MaskData, min_area: int, nms_thresh: float
 337 |     ) -> MaskData:
 338 |         """
 339 |         Removes small disconnected regions and holes in masks, then reruns
 340 |         box NMS to remove any new duplicates.
 341 | 
 342 |         Edits mask_data in place.
 343 | 
 344 |         Requires open-cv as a dependency.
 345 |         """
 346 |         if len(mask_data["rles"]) == 0:
 347 |             return mask_data
 348 | 
 349 |         # Filter small disconnected regions and holes
 350 |         new_masks = []
 351 |         scores = []
 352 |         for rle in mask_data["rles"]:
 353 |             mask = rle_to_mask(rle)
 354 | 
 355 |             mask, changed = remove_small_regions(mask, min_area, mode="holes")
 356 |             unchanged = not changed
 357 |             mask, changed = remove_small_regions(mask, min_area, mode="islands")
 358 |             unchanged = unchanged and not changed
 359 | 
 360 |             new_masks.append(torch.as_tensor(mask).unsqueeze(0))
 361 |             # Give score=0 to changed masks and score=1 to unchanged masks
 362 |             # so NMS will prefer ones that didn't need postprocessing
 363 |             scores.append(float(unchanged))
 364 | 
 365 |         # Recalculate boxes and remove any new duplicates
 366 |         masks = torch.cat(new_masks, dim=0)
 367 |         boxes = batched_mask_to_box(masks)
 368 |         keep_by_nms = batched_nms(
 369 |             boxes.float(),
 370 |             torch.as_tensor(scores),
 371 |             torch.zeros_like(boxes[:, 0]),  # categories
 372 |             iou_threshold=nms_thresh,
 373 |         )
 374 | 
 375 |         # Only recalculate RLEs for masks that have changed
 376 |         for i_mask in keep_by_nms:
 377 |             if scores[i_mask] == 0.0:
 378 |                 mask_torch = masks[i_mask].unsqueeze(0)
 379 |                 mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
 380 |                 mask_data["boxes"][i_mask] = boxes[i_mask]  # update res directly
 381 |         mask_data.filter(keep_by_nms)
 382 | 
 383 |         return mask_data
```
