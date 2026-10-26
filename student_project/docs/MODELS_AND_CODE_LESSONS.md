# Learn the models first, then their code, then the assignment

## Lesson 1: What are we building?
An ultrasound image is an array of pixel intensities. Here it is represented as H x W x 3: height, width and three channels. A model is a mathematical program with learned numerical parameters (weights). Code defines the program; a checkpoint supplies weights. A forward pass applies the program to inputs. Training adjusts weights using examples and a loss.

Two different questions are involved:

| Model | Question | Input | Output | Current weights |
|---|---|---|---|---|
| LiteMedSAM | Which pixels belong to the prompted region? | Image + rectangle | Pixel mask | Official pretrained checkpoint |
| MedNet / Darma | What scores does this region receive? | Resized image crop | Two raw scores, plus feature maps internally | Random initialization for software testing |

A mask has one foreground/background value for every pixel. A crop is a rectangular piece of the original image. They are not interchangeable. The mask determines a tight rectangle; the crop retains the original pixels inside it, including any background. Our first prototype uses unmasked rectangular crops. No meanings such as benign/malignant are assigned to the two output positions.

Example: the image is 420 pixels high and 580 wide. The supplied prompt is [265,78,347,196] in [xmin,ymin,xmax,ymax] order. LiteMedSAM returns a 420x580 mask. The mask's tight rectangle in our local run is [273,90,347,187], using exclusive upper bounds. That crop is 74 pixels wide and 97 high. We resize it to 224x224 before Darma, which returns shape [1,2]. The first dimension is batch size, not a class.

**Check yourself:** Can you explain why the prompt rectangle, mask and tight crop have different shapes? Why do random classification weights not provide a useful disease result?

## Lesson 2: LiteMedSAM's three components
1. Image encoder (TinyViT): converts the image to a feature grid. Features describe image patterns numerically; they are not a viewable segmentation mask.
2. Prompt encoder: converts box corners into vectors and supplies positional information. Without the box, this demonstration does not know which region you intend.
3. Mask decoder: combines image features and prompt information through attention and predicts a mask.

The README describes first distilling the lighter image encoder from a larger MedSAM teacher, then fine-tuning the segmentation pipeline. We use the resulting pretrained checkpoint; we are not repeating training.

Trace the tensor dimensions: HWC image -> longest side 256 -> normalization -> bottom/right padding -> [1,3,256,256] tensor -> [1,256,64,64] image embedding. Two box corners produce [1,2,256] sparse prompt vectors. Decoder logits are [1,1,256,256]. Crop padding, resize logits to 420x580, apply sigmoid, threshold >0.5. Foreground is 1, background is 0.

`eval()` changes dropout/batch-normalization behavior; `inference_mode()` disables gradient tracking. They do different jobs. `to('cuda:0')` moves tensors/modules to GPU; choosing a GPU runtime alone does not do that. The official CLI handles the device argument.

**Read now:** UNDERSTANDING_LITEMEDSAM.md for the block-by-block explanation and SOURCE_INDEX.md for exact source spans.

## Lesson 3: Darma's components, from actual code
MedNet.py contains seven architecture classes:

- ChannelAttention: spatial mean and maximum summaries are passed through a shared small neural network. Sigmoid weights rescale feature channels: which kinds of features matter?
- SpatialAttention: mean and maximum across channels form two spatial maps. A 7x7 convolution and sigmoid weight locations: where do features matter?
- CBAM: applies channel attention, then spatial attention.
- DepthwiseSeparableConv: a 3x3 convolution filters each input channel separately (`groups=in_channels`); a 1x1 convolution mixes channels. Batch normalization and ReLU follow.
- ResidualDSCBAMBlock: two separable convolutions plus optional CBAM, added to a shortcut. The shortcut projects channels/downsamples if needed so addition has compatible shapes.
- ResidualCBAMBackbone: five stages. For 224x224 input, activations are [1,64,224,224], [1,128,112,112], [1,256,56,56], [1,512,28,28], [1,1024,14,14]. It returns the final grid and all five activations.
- Darma: adaptive global average pooling turns the last spatial grid into 1024 numbers. A linear layer maps 1024->256; dropout is used during training; another linear layer maps 256->2. `forward` returns `(scores, activations)`, not just scores. It does not apply softmax.

**Check yourself:** Why does the shortcut sometimes need a 1x1 convolution? What information is removed by average pooling? Why must we unpack `scores, activations = model(x)`?

## Lesson 4: Reading the actual repository, not assuming from its README
LiteMedSAM folder reading order:
1. README.md: branch, setup, checkpoint and demo pointers.
2. CVPR24_LiteMedSAM_infer.py: CLI, preprocessing, constructors, checkpoint loading, 2D inference loop, output saving. Start here because it is the script actually executed.
3. tiny_vit_sam.py: the image encoder; follow the classes described in the LiteMedSAM guide.
4. segment_anything/modeling/prompt_encoder.py: boxes -> embeddings.
5. segment_anything/modeling/transformer.py: information exchange through attention.
6. segment_anything/modeling/mask_decoder.py and common.py: output masks and supporting layers.
7. train_one_gpu.py: dataset, loss, backward and optimizer; read after inference.
8. distillation/: teacher/student encoder training. Distributed training, 3D, preprocessing, evaluation and Docker files are separate paths. They need not all be run for one-image inference.

MedNet folder reading order:
1. MedNet.py architecture classes above, then `forward` methods.
2. Its evaluation transform inside `main`: resize, normalize and tensor conversion.
3. AlbumentationsTransform: surprisingly, it converts RGB to BGR first. Our adapter explicitly reproduces that order, then uses the source's normalization constants. This preserves source behavior; it does not prove that the color convention is scientifically optimal.
4. Training and test loops: scores, losses, optimizer, checkpoint saving, metrics. Do not run the training entry point for the two-score test.
5. preprocessing.py: the actual file reads a BloodMNIST NPZ and exports predefined split images. It is not a generic ultrasound preparation system. Its blood-cell classes do not define our two future study classes.

The README example `from MedNet import MedNet` does not match the actual `Darma` class. We follow the source and the handout. No trained MedNet checkpoint was present in the inspected three-file repository, so the assignment's initial test uses untrained weights explicitly.

## Lesson 5: Our modules — each file has one responsibility
- src/segmentation.py: validate image/box, save demo NPZ, invoke original script, capture log, validate returned mask. The upstream model implementation is not rewritten.
- src/cropping.py: locate nonzero pixels, compute exclusive tight bounds, add each-side margins and clip to image bounds. Empty masks raise an explicit error rather than pretending a valid crop exists.
- src/classification.py: select and compile the seven original architecture class bodies using Python AST; omit training-only imports/functions. Initialize Darma with two outputs; prepare crops; unpack and validate outputs. Original architecture operations are unchanged.
- src/pipeline.py: connect the above and save input, box, mask, three crops, scores, source/checkpoint provenance, logs, versions and wall time. Total time includes loading and file I/O.

This AST adapter is an engineering adjustment, not a newly invented model. The source hash and Git commit are recorded so changes can be audited. Image size 224 is an explicit example configuration supported by the source, not a uniquely mandatory resolution.

## Lesson 6: Do the tasks in order
1. Open notebooks/01_litemedsam_test.ipynb and explain each preprocessing step before running it. This copies the already reviewed Colab notebook.
2. Open notebooks/02_mednet_test.ipynb. Predict all five feature shapes, run and inspect actual scores. State that weights are untrained.
3. Inspect cropping tests. Calculate 10% and 20% margins manually for a rectangle; compare with code.
4. Run notebooks/03_integrated_demo.ipynb and inspect every saved artifact.
5. Explain one actual failure and its resolution using ISSUE_LOG.md. Report exact evidence rather than calling a prepared notebook a successful GPU run.

For every code block, answer: What inputs does it receive? What shape/type/device are they? What operation changes them? What output is returned/saved? Why is that operation needed? What fails if it is omitted? Then explain the line in your own words. This is the process for understanding every line; an automatically numbered source listing alone is not an explanation.

## Professor-facing statement
"We have official pretrained segmentation working on a supplied demo. The classifier architecture executes with two untrained outputs. The integration crops the segmentation and records scores, provenance and timing. These are engineering checks. Trained classification, dataset-specific evaluation and patient-level validation remain future work after the research data and labels are supplied."

Use the latest task audit for which local and Colab checks have actually passed. Student understanding still requires working through the lessons and exercises; no document can certify that on your behalf.
