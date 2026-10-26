# segment_anything/utils/amg.py

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
  11 | import math
  12 | from copy import deepcopy
  13 | from itertools import product
  14 | from typing import Any, Dict, Generator, ItemsView, List, Tuple
  15 | 
  16 | 
  17 | class MaskData:
  18 |     """
  19 |     A structure for storing masks and their related data in batched format.
  20 |     Implements basic filtering and concatenation.
  21 |     """
  22 | 
  23 |     def __init__(self, **kwargs) -> None:
  24 |         for v in kwargs.values():
  25 |             assert isinstance(
  26 |                 v, (list, np.ndarray, torch.Tensor)
  27 |             ), "MaskData only supports list, numpy arrays, and torch tensors."
  28 |         self._stats = dict(**kwargs)
  29 | 
  30 |     def __setitem__(self, key: str, item: Any) -> None:
  31 |         assert isinstance(
  32 |             item, (list, np.ndarray, torch.Tensor)
  33 |         ), "MaskData only supports list, numpy arrays, and torch tensors."
  34 |         self._stats[key] = item
  35 | 
  36 |     def __delitem__(self, key: str) -> None:
  37 |         del self._stats[key]
  38 | 
  39 |     def __getitem__(self, key: str) -> Any:
  40 |         return self._stats[key]
  41 | 
  42 |     def items(self) -> ItemsView[str, Any]:
  43 |         return self._stats.items()
  44 | 
  45 |     def filter(self, keep: torch.Tensor) -> None:
  46 |         for k, v in self._stats.items():
  47 |             if v is None:
  48 |                 self._stats[k] = None
  49 |             elif isinstance(v, torch.Tensor):
  50 |                 self._stats[k] = v[torch.as_tensor(keep, device=v.device)]
  51 |             elif isinstance(v, np.ndarray):
  52 |                 self._stats[k] = v[keep.detach().cpu().numpy()]
  53 |             elif isinstance(v, list) and keep.dtype == torch.bool:
  54 |                 self._stats[k] = [a for i, a in enumerate(v) if keep[i]]
  55 |             elif isinstance(v, list):
  56 |                 self._stats[k] = [v[i] for i in keep]
  57 |             else:
  58 |                 raise TypeError(f"MaskData key {k} has an unsupported type {type(v)}.")
  59 | 
  60 |     def cat(self, new_stats: "MaskData") -> None:
  61 |         for k, v in new_stats.items():
  62 |             if k not in self._stats or self._stats[k] is None:
  63 |                 self._stats[k] = deepcopy(v)
  64 |             elif isinstance(v, torch.Tensor):
  65 |                 self._stats[k] = torch.cat([self._stats[k], v], dim=0)
  66 |             elif isinstance(v, np.ndarray):
  67 |                 self._stats[k] = np.concatenate([self._stats[k], v], axis=0)
  68 |             elif isinstance(v, list):
  69 |                 self._stats[k] = self._stats[k] + deepcopy(v)
  70 |             else:
  71 |                 raise TypeError(f"MaskData key {k} has an unsupported type {type(v)}.")
  72 | 
  73 |     def to_numpy(self) -> None:
  74 |         for k, v in self._stats.items():
  75 |             if isinstance(v, torch.Tensor):
  76 |                 self._stats[k] = v.detach().cpu().numpy()
  77 | 
  78 | 
  79 | def is_box_near_crop_edge(
  80 |     boxes: torch.Tensor, crop_box: List[int], orig_box: List[int], atol: float = 20.0
  81 | ) -> torch.Tensor:
  82 |     """Filter masks at the edge of a crop, but not at the edge of the original image."""
  83 |     crop_box_torch = torch.as_tensor(crop_box, dtype=torch.float, device=boxes.device)
  84 |     orig_box_torch = torch.as_tensor(orig_box, dtype=torch.float, device=boxes.device)
  85 |     boxes = uncrop_boxes_xyxy(boxes, crop_box).float()
  86 |     near_crop_edge = torch.isclose(boxes, crop_box_torch[None, :], atol=atol, rtol=0)
  87 |     near_image_edge = torch.isclose(boxes, orig_box_torch[None, :], atol=atol, rtol=0)
  88 |     near_crop_edge = torch.logical_and(near_crop_edge, ~near_image_edge)
  89 |     return torch.any(near_crop_edge, dim=1)
  90 | 
  91 | 
  92 | def box_xyxy_to_xywh(box_xyxy: torch.Tensor) -> torch.Tensor:
  93 |     box_xywh = deepcopy(box_xyxy)
  94 |     box_xywh[2] = box_xywh[2] - box_xywh[0]
  95 |     box_xywh[3] = box_xywh[3] - box_xywh[1]
  96 |     return box_xywh
  97 | 
  98 | 
  99 | def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
 100 |     assert len(args) > 0 and all(
 101 |         len(a) == len(args[0]) for a in args
 102 |     ), "Batched iteration must have inputs of all the same size."
 103 |     n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
 104 |     for b in range(n_batches):
 105 |         yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]
 106 | 
 107 | 
 108 | def mask_to_rle_pytorch(tensor: torch.Tensor) -> List[Dict[str, Any]]:
 109 |     """
 110 |     Encodes masks to an uncompressed RLE, in the format expected by
 111 |     pycoco tools.
 112 |     """
 113 |     # Put in fortran order and flatten h,w
 114 |     b, h, w = tensor.shape
 115 |     tensor = tensor.permute(0, 2, 1).flatten(1)
 116 | 
 117 |     # Compute change indices
 118 |     diff = tensor[:, 1:] ^ tensor[:, :-1]
 119 |     change_indices = diff.nonzero()
 120 | 
 121 |     # Encode run length
 122 |     out = []
 123 |     for i in range(b):
 124 |         cur_idxs = change_indices[change_indices[:, 0] == i, 1]
 125 |         cur_idxs = torch.cat(
 126 |             [
 127 |                 torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device),
 128 |                 cur_idxs + 1,
 129 |                 torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device),
 130 |             ]
 131 |         )
 132 |         btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
 133 |         counts = [] if tensor[i, 0] == 0 else [0]
 134 |         counts.extend(btw_idxs.detach().cpu().tolist())
 135 |         out.append({"size": [h, w], "counts": counts})
 136 |     return out
 137 | 
 138 | 
 139 | def rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
 140 |     """Compute a binary mask from an uncompressed RLE."""
 141 |     h, w = rle["size"]
 142 |     mask = np.empty(h * w, dtype=bool)
 143 |     idx = 0
 144 |     parity = False
 145 |     for count in rle["counts"]:
 146 |         mask[idx : idx + count] = parity
 147 |         idx += count
 148 |         parity ^= True
 149 |     mask = mask.reshape(w, h)
 150 |     return mask.transpose()  # Put in C order
 151 | 
 152 | 
 153 | def area_from_rle(rle: Dict[str, Any]) -> int:
 154 |     return sum(rle["counts"][1::2])
 155 | 
 156 | 
 157 | def calculate_stability_score(
 158 |     masks: torch.Tensor, mask_threshold: float, threshold_offset: float
 159 | ) -> torch.Tensor:
 160 |     """
 161 |     Computes the stability score for a batch of masks. The stability
 162 |     score is the IoU between the binary masks obtained by thresholding
 163 |     the predicted mask logits at high and low values.
 164 |     """
 165 |     # One mask is always contained inside the other.
 166 |     # Save memory by preventing unnecessary cast to torch.int64
 167 |     intersections = (
 168 |         (masks > (mask_threshold + threshold_offset))
 169 |         .sum(-1, dtype=torch.int16)
 170 |         .sum(-1, dtype=torch.int32)
 171 |     )
 172 |     unions = (
 173 |         (masks > (mask_threshold - threshold_offset))
 174 |         .sum(-1, dtype=torch.int16)
 175 |         .sum(-1, dtype=torch.int32)
 176 |     )
 177 |     return intersections / unions
 178 | 
 179 | 
 180 | def build_point_grid(n_per_side: int) -> np.ndarray:
 181 |     """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
 182 |     offset = 1 / (2 * n_per_side)
 183 |     points_one_side = np.linspace(offset, 1 - offset, n_per_side)
 184 |     points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
 185 |     points_y = np.tile(points_one_side[:, None], (1, n_per_side))
 186 |     points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
 187 |     return points
 188 | 
 189 | 
 190 | def build_all_layer_point_grids(
 191 |     n_per_side: int, n_layers: int, scale_per_layer: int
 192 | ) -> List[np.ndarray]:
 193 |     """Generates point grids for all crop layers."""
 194 |     points_by_layer = []
 195 |     for i in range(n_layers + 1):
 196 |         n_points = int(n_per_side / (scale_per_layer**i))
 197 |         points_by_layer.append(build_point_grid(n_points))
 198 |     return points_by_layer
 199 | 
 200 | 
 201 | def generate_crop_boxes(
 202 |     im_size: Tuple[int, ...], n_layers: int, overlap_ratio: float
 203 | ) -> Tuple[List[List[int]], List[int]]:
 204 |     """
 205 |     Generates a list of crop boxes of different sizes. Each layer
 206 |     has (2**i)**2 boxes for the ith layer.
 207 |     """
 208 |     crop_boxes, layer_idxs = [], []
 209 |     im_h, im_w = im_size
 210 |     short_side = min(im_h, im_w)
 211 | 
 212 |     # Original image
 213 |     crop_boxes.append([0, 0, im_w, im_h])
 214 |     layer_idxs.append(0)
 215 | 
 216 |     def crop_len(orig_len, n_crops, overlap):
 217 |         return int(math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops))
 218 | 
 219 |     for i_layer in range(n_layers):
 220 |         n_crops_per_side = 2 ** (i_layer + 1)
 221 |         overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))
 222 | 
 223 |         crop_w = crop_len(im_w, n_crops_per_side, overlap)
 224 |         crop_h = crop_len(im_h, n_crops_per_side, overlap)
 225 | 
 226 |         crop_box_x0 = [int((crop_w - overlap) * i) for i in range(n_crops_per_side)]
 227 |         crop_box_y0 = [int((crop_h - overlap) * i) for i in range(n_crops_per_side)]
 228 | 
 229 |         # Crops in XYWH format
 230 |         for x0, y0 in product(crop_box_x0, crop_box_y0):
 231 |             box = [x0, y0, min(x0 + crop_w, im_w), min(y0 + crop_h, im_h)]
 232 |             crop_boxes.append(box)
 233 |             layer_idxs.append(i_layer + 1)
 234 | 
 235 |     return crop_boxes, layer_idxs
 236 | 
 237 | 
 238 | def uncrop_boxes_xyxy(boxes: torch.Tensor, crop_box: List[int]) -> torch.Tensor:
 239 |     x0, y0, _, _ = crop_box
 240 |     offset = torch.tensor([[x0, y0, x0, y0]], device=boxes.device)
 241 |     # Check if boxes has a channel dimension
 242 |     if len(boxes.shape) == 3:
 243 |         offset = offset.unsqueeze(1)
 244 |     return boxes + offset
 245 | 
 246 | 
 247 | def uncrop_points(points: torch.Tensor, crop_box: List[int]) -> torch.Tensor:
 248 |     x0, y0, _, _ = crop_box
 249 |     offset = torch.tensor([[x0, y0]], device=points.device)
 250 |     # Check if points has a channel dimension
 251 |     if len(points.shape) == 3:
 252 |         offset = offset.unsqueeze(1)
 253 |     return points + offset
 254 | 
 255 | 
 256 | def uncrop_masks(
 257 |     masks: torch.Tensor, crop_box: List[int], orig_h: int, orig_w: int
 258 | ) -> torch.Tensor:
 259 |     x0, y0, x1, y1 = crop_box
 260 |     if x0 == 0 and y0 == 0 and x1 == orig_w and y1 == orig_h:
 261 |         return masks
 262 |     # Coordinate transform masks
 263 |     pad_x, pad_y = orig_w - (x1 - x0), orig_h - (y1 - y0)
 264 |     pad = (x0, pad_x - x0, y0, pad_y - y0)
 265 |     return torch.nn.functional.pad(masks, pad, value=0)
 266 | 
 267 | 
 268 | def remove_small_regions(
 269 |     mask: np.ndarray, area_thresh: float, mode: str
 270 | ) -> Tuple[np.ndarray, bool]:
 271 |     """
 272 |     Removes small disconnected regions and holes in a mask. Returns the
 273 |     mask and an indicator of if the mask has been modified.
 274 |     """
 275 |     import cv2  # type: ignore
 276 | 
 277 |     assert mode in ["holes", "islands"]
 278 |     correct_holes = mode == "holes"
 279 |     working_mask = (correct_holes ^ mask).astype(np.uint8)
 280 |     n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
 281 |     sizes = stats[:, -1][1:]  # Row 0 is background label
 282 |     small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
 283 |     if len(small_regions) == 0:
 284 |         return mask, False
 285 |     fill_labels = [0] + small_regions
 286 |     if not correct_holes:
 287 |         fill_labels = [i for i in range(n_labels) if i not in fill_labels]
 288 |         # If every region is below threshold, keep largest
 289 |         if len(fill_labels) == 0:
 290 |             fill_labels = [int(np.argmax(sizes)) + 1]
 291 |     mask = np.isin(regions, fill_labels)
 292 |     return mask, True
 293 | 
 294 | 
 295 | def coco_encode_rle(uncompressed_rle: Dict[str, Any]) -> Dict[str, Any]:
 296 |     from pycocotools import mask as mask_utils  # type: ignore
 297 | 
 298 |     h, w = uncompressed_rle["size"]
 299 |     rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
 300 |     rle["counts"] = rle["counts"].decode("utf-8")  # Necessary to serialize with json
 301 |     return rle
 302 | 
 303 | 
 304 | def batched_mask_to_box(masks: torch.Tensor) -> torch.Tensor:
 305 |     """
 306 |     Calculates boxes in XYXY format around masks. Return [0,0,0,0] for
 307 |     an empty mask. For input shape C1xC2x...xHxW, the output shape is C1xC2x...x4.
 308 |     """
 309 |     # torch.max below raises an error on empty inputs, just skip in this case
 310 |     if torch.numel(masks) == 0:
 311 |         return torch.zeros(*masks.shape[:-2], 4, device=masks.device)
 312 | 
 313 |     # Normalize shape to CxHxW
 314 |     shape = masks.shape
 315 |     h, w = shape[-2:]
 316 |     if len(shape) > 2:
 317 |         masks = masks.flatten(0, -3)
 318 |     else:
 319 |         masks = masks.unsqueeze(0)
 320 | 
 321 |     # Get top and bottom edges
 322 |     in_height, _ = torch.max(masks, dim=-1)
 323 |     in_height_coords = in_height * torch.arange(h, device=in_height.device)[None, :]
 324 |     bottom_edges, _ = torch.max(in_height_coords, dim=-1)
 325 |     in_height_coords = in_height_coords + h * (~in_height)
 326 |     top_edges, _ = torch.min(in_height_coords, dim=-1)
 327 | 
 328 |     # Get left and right edges
 329 |     in_width, _ = torch.max(masks, dim=-2)
 330 |     in_width_coords = in_width * torch.arange(w, device=in_width.device)[None, :]
 331 |     right_edges, _ = torch.max(in_width_coords, dim=-1)
 332 |     in_width_coords = in_width_coords + w * (~in_width)
 333 |     left_edges, _ = torch.min(in_width_coords, dim=-1)
 334 | 
 335 |     # If the mask is empty the right edge will be to the left of the left edge.
 336 |     # Replace these boxes with [0, 0, 0, 0]
 337 |     empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
 338 |     out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
 339 |     out = out * (~empty_filter).unsqueeze(-1)
 340 | 
 341 |     # Return to original shape
 342 |     if len(shape) > 2:
 343 |         out = out.reshape(*shape[:-2], 4)
 344 |     else:
 345 |         out = out[0]
 346 | 
 347 |     return out
```
