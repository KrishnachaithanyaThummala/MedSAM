# Understanding LiteMedSAM

## Start with the task
Segmentation means assigning a foreground/background decision to each pixel. A bounding box tells this model which region you want. It does not identify a disease or automatically discover a lesion. Your professor's handout describes a later classification pipeline; the current workspace concentrates on LiteMedSAM.

The main flow is:
RGB image + box -> resize/normalize/pad -> TinyViT image features + encoded box -> two-way transformer -> mask logits -> crop padding/restore image size -> sigmoid/threshold -> binary mask.

## Vocabulary before reading code
- Tensor: a multidimensional array. B,C,H,W means batch, channels, height, width.
- `nn.Module`: a trainable component. `__init__` creates layers; `forward` applies them.
- Parameters/weights: learned numerical values. Architecture alone gives untrained outputs; the checkpoint supplies learned weights.
- Embedding: learned numerical representation of an image or prompt.
- Logit: raw output before sigmoid. Sigmoid maps it into (0,1); thresholding at 0.5 produces a binary mask.
- Attention: weighted information exchange between tokens. Tokens can represent image locations or prompts.
- `model.eval()`: changes training-dependent layers such as batch normalization/dropout. It does not itself disable gradient recording.
- `torch.inference_mode()`: disables gradient tracking for inference. Training requires gradients.
- `permute(2,0,1)`: converts HWC image layout to CHW. `unsqueeze(0)` adds the batch dimension.

## A concrete shape trace
For a single RGB image, resize its longest side to 256, retaining aspect ratio. Normalize using the resized image minimum and maximum. Pad the bottom/right to 256x256. Input is float32 [1,3,256,256].

The image encoder returns [1,256,64,64]. Here 256 means feature channels, not image width. A box [xmin,ymin,xmax,ymax] is uniformly scaled into the resized coordinate system. The prompt encoder represents its two corners as [1,2,256] sparse embeddings. With no mask prompt, a learned no-mask embedding is expanded into [1,256,64,64].

The decoder combines image and prompt information, predicts [1,1,256,256] mask logits and [1,1] predicted mask quality. Remove padding from logits, interpolate back to original H,W, then apply sigmoid and >0.5. Interpolating logits before thresholding preserves smooth boundaries. The predicted quality is a learned estimate, not measured IoU and not disease confidence.

## Read these files in this order
Line references below belong to upstream commit b0fab476e54e631dd412b25e0db9fdf2a2b0f54c. The generated SOURCE_INDEX.md contains exact function spans and line-numbered source pages for every upstream Python file. Those pages are navigation aids, not a claim that every line has already been explained or mastered.

1. `scripts/run_litemedsam.py`: our small single-image wrapper. Read each assignment while inspecting notebook outputs. It uses original upstream component implementations. It avoids importing scripts that parse arguments and load models at import time.
2. `CVPR24_LiteMedSAM_infer.py`: upstream challenge inference. Inspect resize/pad helpers, `MedSAM_Lite`, box conversion, `medsam_inference`, constructors and the 2D input loop. This script accepts boxes in NPZ files.
3. `tiny_vit_sam.py`: the lightweight image encoder. Read the blocks below, then trace `TinyViT.forward_features` at 608 and `forward` at 626.
4. `segment_anything/modeling/prompt_encoder.py`: box corner positions, sparse/dense embeddings and positional encodings.
5. `segment_anything/modeling/transformer.py`: the interaction between image features and prompt/output tokens.
6. `segment_anything/modeling/mask_decoder.py`: output tokens, upscaling and mask generation.
7. `train_one_gpu.py`, then `distillation/distill_single_gpu.py`: how weights are learned.
8. Preprocessing, 3D, distributed training and evaluation files after the single-image path is clear.

## TinyViT block by block
`Conv2d_BN` (23): convolution without bias followed by batch normalization. `fuse` algebraically combines their inference operations using running statistics. This helper is separate from the main forward path.

`PatchEmbed` (59): in THIS adaptation two 1x1 convolutions project RGB into 64 channels without spatial downsampling. Do not assume standard TinyViT's original patch reduction: the old stride-2 code is commented out.

`MBConv` (85): expand channels with 1x1 convolution, depthwise 3x3 spatial filtering, project back, add shortcut, activate. A depthwise convolution filters channels independently. The shortcut helps preserve information and gradients.

`PatchMerging` (126): maps channels and generally downsamples with depthwise stride 2. For out_dim=320 the stride is 1. Thus this model retains a 64x64 final feature grid.

`ConvLayer` (158): runs a sequence of MBConv blocks, optionally using gradient checkpointing during training, and optionally merges patches.

`Mlp` (197): layer normalization, linear projection, activation, linear projection and dropout. It transforms each token's channel representation.

`Attention` (220): creates queries, keys and values; computes scaled query-key similarities plus learned spatial biases; softmax produces weights; weighted values are projected back. Evaluation caches attention biases, so `eval()` matters.

`TinyViTBlock` (295): partitions tokens into local windows, pads to accommodate window size, applies attention, restores the grid, applies local convolution and MLP with residual connections. Reshape changes interpretation; permute changes axis order. Track both before flattening again.

`BasicLayer` (391): stacks TinyViT blocks and optionally merges patches. Window attention limits the cost compared with comparing every image location to every other location.

`LayerNorm2d` (457): normalize each pixel across feature channels and apply learned scale/bias.

`TinyViT` (471): constructs all stages and the neck. In `forward_features`, the last tokens are reshaped to [B,64,64,C], permuted to BCHW, and the neck produces 256 channels. The hard-coded 64x64 shape means arbitrary input resolutions are not automatically supported.

## Prompt encoder
`_embed_boxes` adds 0.5 to coordinates, treating positions as pixel centers; reshapes boxes into pairs of corners; creates positional embeddings and adds different learned embeddings for the first and second corners. `points=None` means no additional point prompts. `masks=None` means no input mask; the learned no-mask embedding is used. `get_dense_pe()` returns positional information for the 64x64 image grid. Position encodings preserve location information that attention alone would not provide.

## Decoder and two-way attention
`MaskDecoder.__init__` creates an IoU token, four mask tokens (one single-mask token plus three ambiguity tokens), two transposed convolutions, per-mask small MLPs and a quality head.

`predict_masks` concatenates output tokens with prompt tokens; adds dense prompt embeddings to image embeddings; runs the two-way transformer. Tokens attend to tokens, tokens attend to the image, and the image attends back to tokens. The final image features are upscaled from 64x64 to 256x256. A small MLP transforms each mask token into coefficients that combine those upscaled features into a mask. The quality token feeds a separate MLP.

`multimask_output=False` selects mask token zero. Although the constructor has num_multimask_outputs=3, it does not return three masks in our run.

## Training versus inference
The README describes encoder distillation from the larger MedSAM encoder, followed by segmentation fine-tuning. Distillation teaches the student encoder to reproduce teacher features; fine-tuning teaches the image+prompt pipeline to predict masks.

In `train_one_gpu.py`, inspect the dataset class, random foreground selection, box perturbation, model forward, AdamW setup and training loop. The implemented loss is weighted Dice loss + binary cross entropy with logits + MSE between predicted and target mask IoU. `zero_grad` clears gradients; `backward` computes them; `step` updates parameters. Training and validation must use appropriate separate samples; a demonstration forward pass proves none of the clinical performance claims.

Do not execute training as a prerequisite for inference: we already have official pretrained weights. The installed environment covers inference/notebooks; full training needs additional dependencies such as MONAI and suitable data/compute.

## Code observations to understand
- `inference_2D.py` derives boxes from ground-truth masks. Its results assume that prompt information is available. Challenge inference instead reads supplied boxes.
- In `inference_2D.py`'s `MedSAM_Lite.forward`, a converted `box_torch` is created but `boxes=box_np` is passed onward. The separate `medsam_inference` helper uses the converted tensor. Our runner directly uses tensor boxes.
- Several scripts parse CLI arguments and instantiate models at module scope, so importing them as libraries has side effects. Our wrapper imports component classes instead.
- Upstream box conversion truncates after scaling. Our wrapper matches that behavior; half-open image bounds are accepted by wrapper validation.
- Later masks overwrite earlier labels if upstream multi-box masks overlap. Our demonstration processes only the first supplied box and records it explicitly.
- A constant image remains numerically safe because the normalization denominator is clipped, but this does not establish useful segmentation.

## Study sessions and self-checks
1. Explain segmentation, box coordinates, and the meaning of each saved output. Run the notebook yourself.
2. Explain every line of our small runner; predict every intermediate tensor shape before executing.
3. Walk through each TinyViT block above with its numbered source. Explain why this adaptation preserves a 64x64 grid.
4. Trace a box through `_embed_boxes`, then through both directions of transformer attention and the decoder.
5. Trace dataset -> loss -> gradients -> optimizer -> checkpoint in training; then compare encoder distillation.
6. Inspect remaining scripts using the complete source index. Explain which are relevant to your own pipeline.

Completion means you can change a box, predict how coordinate conversion changes, rerun and interpret the differences; identify every trainable component; explain where weights originate; and distinguish an execution check from performance evaluation. Reading a guide once does not establish line-by-line understanding. Use it as the basis for an interactive file-by-file walkthrough.
