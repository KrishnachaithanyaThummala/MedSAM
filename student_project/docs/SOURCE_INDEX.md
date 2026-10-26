# Complete upstream Python source index

Commit: b0fab476e54e631dd412b25e0db9fdf2a2b0f54c

Generated exact line spans and numbered source; explanations are in the study guide.


## [CVPR24_LiteMedSAM_infer.py](source/CVPR24_LiteMedSAM_infer.py.md)

- `resize_longest_side`: lines 91-102
- `pad_image`: lines 104-118
- `MedSAM_Lite`: lines 120-184
- `show_mask`: lines 187-208
- `show_box`: lines 211-226
- `get_bbox256`: lines 229-259
- `resize_box_to_256`: lines 261-283
- `medsam_inference`: lines 287-320
- `MedSAM_infer_npz_2D`: lines 374-424
- `MedSAM_infer_npz_3D`: lines 427-541
- `__init__`: lines 121-130
- `forward`: lines 132-153
- `postprocess_masks`: lines 156-184

## [CVPR24_time_eval.py](source/CVPR24_time_eval.py.md)


## [LLD_MMRI_step0_data_preprocess.py](source/LLD_MMRI_step0_data_preprocess.py.md)

- `renames`: lines 45-70

## [LLD_MMRI_step1_LiteMedSAM_Infer_MR_LiverCancer.py](source/LLD_MMRI_step1_LiteMedSAM_Infer_MR_LiverCancer.py.md)

- `resize_longest_side`: lines 53-64
- `pad_image`: lines 66-79
- `MedSAM_Lite`: lines 81-145
- `show_mask`: lines 148-155
- `show_box`: lines 158-161
- `resize_box`: lines 164-187
- `medsam_inference`: lines 191-212
- `get_bbox`: lines 214-227
- `infer_MR_liver_tumor`: lines 285-340
- `__init__`: lines 82-91
- `forward`: lines 93-114
- `postprocess_masks`: lines 117-145

## [distillation/distill_multi_gpu.py](source/distillation__distill_multi_gpu.py.md)

- `NpyDataset`: lines 36-82
- `Logger`: lines 85-129
- `get_args`: lines 132-166
- `revert_sync_batchnorm`: lines 169-192
- `main`: lines 195-198
- `main_worker`: lines 201-403
- `__init__`: lines 37-44
- `__len__`: lines 46-47
- `__getitem__`: lines 49-82
- `__init__`: lines 86-92
- `log`: lines 94-95
- `plot_progress_png`: lines 97-123
- `get_checkpoint`: lines 125-126
- `load_checkpoint`: lines 128-129

## [distillation/distill_single_gpu.py](source/distillation__distill_single_gpu.py.md)

- `Logger`: lines 69-113
- `NpyDataset`: lines 116-159
- `__init__`: lines 70-76
- `log`: lines 78-79
- `plot_progress_png`: lines 81-107
- `get_checkpoint`: lines 109-110
- `load_checkpoint`: lines 112-113
- `__init__`: lines 117-124
- `__len__`: lines 126-127
- `__getitem__`: lines 129-159

## [distillation/tiny_vit_sam.py](source/distillation__tiny_vit_sam.py.md)

- `Conv2d_BN`: lines 21-43
- `DropPath`: lines 46-54
- `PatchEmbed`: lines 57-80
- `MBConv`: lines 83-121
- `PatchMerging`: lines 124-153
- `ConvLayer`: lines 156-192
- `Mlp`: lines 195-215
- `Attention`: lines 218-290
- `TinyViTBlock`: lines 293-386
- `BasicLayer`: lines 389-453
- `LayerNorm2d`: lines 455-467
- `TinyViT`: lines 469-626
- `__init__`: lines 22-30
- `fuse`: lines 33-43
- `__init__`: lines 47-49
- `__repr__`: lines 51-54
- `__init__`: lines 58-77
- `forward`: lines 79-80
- `__init__`: lines 84-103
- `forward`: lines 105-121
- `__init__`: lines 125-137
- `forward`: lines 139-153
- `__init__`: lines 157-182
- `forward`: lines 184-192
- `__init__`: lines 196-205
- `forward`: lines 207-215
- `__init__`: lines 219-254
- `train`: lines 257-264
- `forward`: lines 266-290
- `__init__`: lines 309-339
- `forward`: lines 341-382
- `extra_repr`: lines 384-386
- `__init__`: lines 408-440
- `forward`: lines 442-450
- `extra_repr`: lines 452-453
- `__init__`: lines 456-460
- `forward`: lines 462-467
- `__init__`: lines 470-560
- `set_layer_lr_decay`: lines 562-591
- `_init_weights`: lines 593-600
- `no_weight_decay_keywords`: lines 603-604
- `forward_features`: lines 606-622
- `forward`: lines 624-626
- `_set_lr_scale`: lines 569-571
- `_check_lr_scale`: lines 587-589

## [evaluation/SurfaceDice.py](source/evaluation__SurfaceDice.py.md)

- `compute_surface_distances`: lines 280-427
- `compute_average_surface_distance`: lines 430-437
- `compute_robust_hausdorff`: lines 439-458
- `compute_surface_overlap_at_tolerance`: lines 460-467
- `compute_surface_dice_at_tolerance`: lines 469-478
- `compute_dice_coefficient`: lines 481-498

## [evaluation/compute_metrics.py](source/evaluation__compute_metrics.py.md)

- `compute_multi_class_dsc`: lines 14-20
- `compute_multi_class_nsd`: lines 22-31
- `compute_metrics`: lines 48-71

## [inference_2D.py](source/inference_2D.py.md)

- `resize_longest_side`: lines 96-107
- `pad_image`: lines 109-122
- `MedSAM_Lite`: lines 125-189
- `show_mask`: lines 192-199
- `show_box`: lines 202-205
- `revert_box`: lines 208-231
- `medsam_inference`: lines 235-258
- `get_bbox`: lines 260-273
- `MedSAM_infer_npz`: lines 329-405
- `__init__`: lines 126-135
- `forward`: lines 137-158
- `postprocess_masks`: lines 161-189

## [inference_3D.py](source/inference_3D.py.md)

- `resize_longest_side`: lines 94-105
- `pad_image`: lines 107-120
- `MedSAM_Lite`: lines 122-186
- `show_mask`: lines 189-196
- `show_box`: lines 199-202
- `resize_box`: lines 205-228
- `medsam_inference`: lines 232-253
- `get_bbox`: lines 255-268
- `MedSAM_infer_npz`: lines 324-400
- `__init__`: lines 123-132
- `forward`: lines 134-155
- `postprocess_masks`: lines 158-186

## [npz_to_npy.py](source/npz_to_npy.py.md)

- `convert_npz_to_npy`: lines 31-72

## [pre_CT_MR.py](source/pre_CT_MR.py.md)

- `preprocess`: lines 86-180

## [pre_grey_rgb.py](source/pre_grey_rgb.py.md)


## [segment_anything/__init__.py](source/segment_anything____init__.py.md)


## [segment_anything/automatic_mask_generator.py](source/segment_anything__automatic_mask_generator.py.md)

- `SamAutomaticMaskGenerator`: lines 36-383
- `__init__`: lines 37-135
- `generate`: lines 138-198
- `_generate_masks`: lines 200-226
- `_process_crop`: lines 228-269
- `_process_batch`: lines 271-332
- `postprocess_small_regions`: lines 335-383

## [segment_anything/build_sam.py](source/segment_anything__build_sam.py.md)

- `build_sam_vit_h`: lines 21-28
- `build_sam_vit_l`: lines 34-41
- `build_sam_vit_b`: lines 44-51
- `_build_sam`: lines 62-146

## [segment_anything/modeling/__init__.py](source/segment_anything__modeling____init__.py.md)


## [segment_anything/modeling/common.py](source/segment_anything__modeling__common.py.md)

- `MLPBlock`: lines 14-27
- `LayerNorm2d`: lines 32-44
- `__init__`: lines 15-24
- `forward`: lines 26-27
- `__init__`: lines 33-37
- `forward`: lines 39-44

## [segment_anything/modeling/image_encoder.py](source/segment_anything__modeling__image_encoder.py.md)

- `ImageEncoderViT`: lines 18-119
- `Block`: lines 122-187
- `Attention`: lines 190-254
- `window_partition`: lines 257-282
- `window_unpartition`: lines 285-312
- `get_rel_pos`: lines 315-345
- `add_decomposed_rel_pos`: lines 348-386
- `PatchEmbed`: lines 389-420
- `__init__`: lines 19-107
- `forward`: lines 109-119
- `__init__`: lines 125-169
- `forward`: lines 171-187
- `__init__`: lines 193-227
- `forward`: lines 229-254
- `__init__`: lines 394-414
- `forward`: lines 416-420

## [segment_anything/modeling/mask_decoder.py](source/segment_anything__modeling__mask_decoder.py.md)

- `MaskDecoder`: lines 17-163
- `MLP`: lines 168-190
- `__init__`: lines 18-74
- `forward`: lines 76-115
- `predict_masks`: lines 117-163
- `__init__`: lines 169-183
- `forward`: lines 185-190

## [segment_anything/modeling/prompt_encoder.py](source/segment_anything__modeling__prompt_encoder.py.md)

- `PromptEncoder`: lines 17-180
- `PositionEmbeddingRandom`: lines 183-226
- `__init__`: lines 18-66
- `get_dense_pe`: lines 68-77
- `_embed_points`: lines 79-99
- `_embed_boxes`: lines 101-110
- `_embed_masks`: lines 112-115
- `_get_batch_size`: lines 117-133
- `_get_device`: lines 135-136
- `forward`: lines 138-180
- `__init__`: lines 188-195
- `_pe_encoding`: lines 197-204
- `forward`: lines 206-217
- `forward_with_coords`: lines 219-226

## [segment_anything/modeling/sam.py](source/segment_anything__modeling__sam.py.md)

- `Sam`: lines 19-181
- `__init__`: lines 23-50
- `device`: lines 53-54
- `forward`: lines 57-136
- `postprocess_masks`: lines 138-169
- `preprocess`: lines 171-181

## [segment_anything/modeling/transformer.py](source/segment_anything__modeling__transformer.py.md)

- `TwoWayTransformer`: lines 17-107
- `TwoWayAttentionBlock`: lines 110-183
- `Attention`: lines 186-243
- `__init__`: lines 18-61
- `forward`: lines 63-107
- `__init__`: lines 111-150
- `forward`: lines 152-183
- `__init__`: lines 192-209
- `_separate_heads`: lines 211-214
- `_recombine_heads`: lines 216-219
- `forward`: lines 221-243

## [segment_anything/predictor.py](source/segment_anything__predictor.py.md)

- `SamPredictor`: lines 18-286
- `__init__`: lines 19-33
- `set_image`: lines 35-63
- `set_torch_image`: lines 66-93
- `predict`: lines 95-177
- `predict_torch`: lines 180-258
- `get_image_embedding`: lines 260-273
- `device`: lines 276-277
- `reset_image`: lines 279-286

## [segment_anything/utils/__init__.py](source/segment_anything__utils____init__.py.md)


## [segment_anything/utils/amg.py](source/segment_anything__utils__amg.py.md)

- `MaskData`: lines 17-76
- `is_box_near_crop_edge`: lines 79-89
- `box_xyxy_to_xywh`: lines 92-96
- `batch_iterator`: lines 99-105
- `mask_to_rle_pytorch`: lines 108-136
- `rle_to_mask`: lines 139-150
- `area_from_rle`: lines 153-154
- `calculate_stability_score`: lines 157-177
- `build_point_grid`: lines 180-187
- `build_all_layer_point_grids`: lines 190-198
- `generate_crop_boxes`: lines 201-235
- `uncrop_boxes_xyxy`: lines 238-244
- `uncrop_points`: lines 247-253
- `uncrop_masks`: lines 256-265
- `remove_small_regions`: lines 268-292
- `coco_encode_rle`: lines 295-301
- `batched_mask_to_box`: lines 304-347
- `__init__`: lines 23-28
- `__setitem__`: lines 30-34
- `__delitem__`: lines 36-37
- `__getitem__`: lines 39-40
- `items`: lines 42-43
- `filter`: lines 45-58
- `cat`: lines 60-71
- `to_numpy`: lines 73-76
- `crop_len`: lines 216-217

## [segment_anything/utils/onnx.py](source/segment_anything__utils__onnx.py.md)

- `SamOnnxModel`: lines 18-158
- `__init__`: lines 26-40
- `resize_longest_image_size`: lines 43-50
- `_embed_points`: lines 52-73
- `_embed_masks`: lines 75-84
- `mask_postprocessing`: lines 86-104
- `select_masks`: lines 106-119
- `forward`: lines 122-158

## [segment_anything/utils/transforms.py](source/segment_anything__utils__transforms.py.md)

- `ResizeLongestSide`: lines 17-111
- `__init__`: lines 24-25
- `apply_image`: lines 27-34
- `apply_coords`: lines 36-48
- `apply_boxes`: lines 50-58
- `apply_image_torch`: lines 60-72
- `apply_coords_torch`: lines 74-88
- `apply_boxes_torch`: lines 90-98
- `get_preprocess_shape`: lines 101-111

## [setup.py](source/setup.py.md)


## [tiny_vit_sam.py](source/tiny_vit_sam.py.md)

- `Conv2d_BN`: lines 23-45
- `DropPath`: lines 48-56
- `PatchEmbed`: lines 59-82
- `MBConv`: lines 85-123
- `PatchMerging`: lines 126-155
- `ConvLayer`: lines 158-194
- `Mlp`: lines 197-217
- `Attention`: lines 220-292
- `TinyViTBlock`: lines 295-388
- `BasicLayer`: lines 391-455
- `LayerNorm2d`: lines 457-469
- `TinyViT`: lines 471-628
- `__init__`: lines 24-32
- `fuse`: lines 35-45
- `__init__`: lines 49-51
- `__repr__`: lines 53-56
- `__init__`: lines 60-79
- `forward`: lines 81-82
- `__init__`: lines 86-105
- `forward`: lines 107-123
- `__init__`: lines 127-139
- `forward`: lines 141-155
- `__init__`: lines 159-184
- `forward`: lines 186-194
- `__init__`: lines 198-207
- `forward`: lines 209-217
- `__init__`: lines 221-256
- `train`: lines 259-266
- `forward`: lines 268-292
- `__init__`: lines 311-341
- `forward`: lines 343-384
- `extra_repr`: lines 386-388
- `__init__`: lines 410-442
- `forward`: lines 444-452
- `extra_repr`: lines 454-455
- `__init__`: lines 458-462
- `forward`: lines 464-469
- `__init__`: lines 472-562
- `set_layer_lr_decay`: lines 564-593
- `_init_weights`: lines 595-602
- `no_weight_decay_keywords`: lines 605-606
- `forward_features`: lines 608-624
- `forward`: lines 626-628
- `_set_lr_scale`: lines 571-573
- `_check_lr_scale`: lines 589-591

## [train_multi_gpus.py](source/train_multi_gpus.py.md)

- `get_args`: lines 37-67
- `show_mask`: lines 70-77
- `show_box`: lines 80-83
- `cal_iou`: lines 87-94
- `revert_sync_batchnorm`: lines 97-120
- `NpyDataset`: lines 123-219
- `collate_fn`: lines 221-232
- `sanity_check_dataset`: lines 235-273
- `MedSAM_Lite`: lines 276-320
- `main`: lines 322-325
- `main_worker`: lines 328-543
- `__init__`: lines 124-133
- `__len__`: lines 135-136
- `__getitem__`: lines 138-191
- `resize_longest_side`: lines 193-204
- `pad_image`: lines 206-219
- `__init__`: lines 277-285
- `forward`: lines 287-303
- `postprocess_masks`: lines 306-320

## [train_one_gpu.py](source/train_one_gpu.py.md)

- `show_mask`: lines 116-123
- `show_box`: lines 125-128
- `cal_iou`: lines 130-137
- `NpyDataset`: lines 140-238
- `MedSAM_Lite`: lines 278-322
- `__init__`: lines 141-153
- `__len__`: lines 155-156
- `__getitem__`: lines 158-210
- `resize_longest_side`: lines 212-223
- `pad_image`: lines 225-238
- `__init__`: lines 279-287
- `forward`: lines 289-305
- `postprocess_masks`: lines 308-322

## [utils/SurfaceDice.py](source/utils__SurfaceDice.py.md)

- `compute_surface_distances`: lines 561-722
- `compute_average_surface_distance`: lines 725-736
- `compute_robust_hausdorff`: lines 739-762
- `compute_surface_overlap_at_tolerance`: lines 765-776
- `compute_surface_dice_at_tolerance`: lines 779-789
- `compute_dice_coefficient`: lines 792-809

## [utils/ckpt_convert.py](source/utils__ckpt_convert.py.md)


## [utils/demo.py](source/utils__demo.py.md)

- `show_mask`: lines 15-22
- `BboxPromptDemo`: lines 25-199
- `PointPromptDemo`: lines 203-348
- `TextPromptDemo`: lines 351-532
- `__init__`: lines 26-37
- `_show`: lines 39-135
- `show`: lines 137-139
- `set_image_path`: lines 141-144
- `_set_image`: lines 146-151
- `_preprocess_image`: lines 153-165
- `_infer`: lines 168-199
- `__init__`: lines 204-218
- `show_mask`: lines 220-227
- `infer`: lines 230-263
- `show`: lines 265-309
- `set_image`: lines 311-328
- `preprocess_image`: lines 330-342
- `get_label_id`: lines 344-348
- `__init__`: lines 352-388
- `show_mask`: lines 390-397
- `infer`: lines 400-425
- `show`: lines 427-483
- `guess_gt_label_id`: lines 485-490
- `set_image`: lines 492-509
- `preprocess_image`: lines 511-523
- `tokenize_text`: lines 526-532
- `__on_press`: lines 52-63
- `__on_release`: lines 65-86
- `__on_motion`: lines 88-101
- `__on_clear_button_clicked`: lines 104-113
- `__on_save_button_clicked`: lines 116-123
- `onclick`: lines 284-306
- `callback`: lines 459-481

## [utils/extract_weights.py](source/utils__extract_weights.py.md)


## [utils/format_convert.py](source/utils__format_convert.py.md)

- `dcm2nii`: lines 10-18
- `mhd2nii`: lines 20-25
- `nii2nii`: lines 27-32
- `nrrd2nii`: lines 34-39
- `jpg2png`: lines 41-46
- `patchfy`: lines 48-92
- `rle_decode`: lines 95-112

## [utils/pre_CT_MR.py](source/utils__pre_CT_MR.py.md)


## [utils/pre_grey_rgb.py](source/utils__pre_grey_rgb.py.md)


## [utils/split.py](source/utils__split.py.md)
