# LiteMedSAM local learning workspace

Scope: understand both models, reproduce LiteMedSAM segmentation, verify an untrained MedNet forward pass, and demonstrate their integration. See the current assignment audit for local versus Colab verification.

- Your fork: https://github.com/KrishnachaithanyaThummala/MedSAM/tree/LiteMedSAM
- Original source: https://github.com/bowang-lab/MedSAM/tree/LiteMedSAM
- Source commit: b0fab476e54e631dd412b25e0db9fdf2a2b0f54c
- Local source: upstream-LiteMedSAM (separate Git checkout; origin is your fork, upstream is bowang-lab).
- Local environment: Python 3.12 on Apple Silicon, CPU inference. Upstream documents Python 3.10/PyTorch 2.1.2/Ubuntu/CUDA; this local environment is a separately tested inference setup.

## Begin here
Read [the study guide](docs/UNDERSTANDING_LITEMEDSAM.md), then open [the executed teaching notebook](notebooks/01_litemedsam_walkthrough.ipynb). [The complete source index](docs/SOURCE_INDEX.md) links line-numbered copies of upstream Python files.

```bash
cd /Users/krishnachaithanyathummala/Documents/ChatGPT/MEDSAM
source .venv/bin/activate
python -m jupyterlab notebooks/01_litemedsam_walkthrough.ipynb
```

For a different RGB image and human-supplied pixel box:

```bash
.venv/bin/python scripts/run_litemedsam.py --image /absolute/path/image.png --box 20 30 120 160
```

Coordinates are xmin ymin xmax ymax. The example numbers must be replaced with a valid box for your image.

## Reproduce installation

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
git clone --branch LiteMedSAM --single-branch https://github.com/bowang-lab/MedSAM.git upstream-LiteMedSAM
git -C upstream-LiteMedSAM checkout b0fab476e54e631dd412b25e0db9fdf2a2b0f54c
mkdir -p checkpoints demo
.venv/bin/python -m gdown 18Zed-TUTsmr2zc5CHUWd5Tu13nb6vq6z -O checkpoints/lite_medsam.pth
.venv/bin/python -m gdown --folder https://drive.google.com/drive/folders/1t3Rs9QbfGSEv2fIFlk8vi7jc0SclD1cq -O demo --remaining-ok
```

The checkpoint and demo links come from the upstream README. No clinical dataset was collected. Checkpoint SHA256: 79d8c9dca6db4d69d3f905579e5250af05e859fff9c1f543e89a513c3028ce76. Download helper can be rerun if a transfer fails.

## Verified result and boundaries
The small teaching runner loaded the official weights strictly, processed the official ultrasound demo's first supplied box [265,78,347,196], and saved a 420x580 binary mask, image, overlay and JSON report under outputs/ultrasound_demo. Foreground: 4,990 pixels. One measured CPU forward pass took approximately 0.412 seconds excluding loading/preprocessing; timing varies. All recorded intermediate tensors were finite. See outputs/ultrasound_demo/run.json for exact observations.

Original component source is unchanged. Our additions are an import-safe single-image wrapper, notebook, study guide and source index. It matches upstream model constructor, image normalization, padding, box truncation and postprocessing. Full training dependencies are not installed. No model training, research-dataset evaluation or clinical performance validation has been performed. Subsequent MedNet and integration work is documented below.

The unmodified upstream CVPR24_LiteMedSAM_infer.py also completed on the same ultrasound input. Its mask and the teaching runner mask are pixel-identical. All four notebook code cells executed successfully and contain saved outputs.

## Google Colab GPU
Upload `notebooks/02_LiteMedSAM_Colab_GPU.ipynb` into Google Colab. Select Runtime > Change runtime type > GPU, connect, then execute cells in order. The notebook clones your fork at the inspected commit, retains Colab's PyTorch/CUDA installation, downloads official weights and one ultrasound demo, runs the original inference entry point with `-device cuda:0`, displays outputs and downloads a results ZIP. Notebook structure and Python syntax have been validated locally; Colab GPU execution is pending. The assignment notebooks are also packaged under student_project on the project branch.

## Full professor assignment: model understanding and integration

Start with [the models-and-code lessons](docs/MODELS_AND_CODE_LESSONS.md), then [the current task audit](docs/ASSIGNMENT_STATUS.md) and [issue log](docs/ISSUE_LOG.md). Earlier sections describe the original LiteMedSAM-only milestone; the project now additionally implements the local untrained MedNet test and integration.

Assignment notebooks (upload into Colab, choose GPU, execute in order):
1. `notebooks/01_litemedsam_test.ipynb` — pretrained segmentation; GPU run previously reviewed.
2. `notebooks/02_mednet_test.ipynb` — Darma two-output test; prepared for GPU; model verified in the existing live Colab notebook.
3. `notebooks/03_integrated_demo.ipynb` — segmentation, three crops and untrained scores; prepared for GPU; model verified in the existing live Colab notebook.

New notebooks clone the project additions from your fork's `codex/student-foundation` branch under `student_project/`. Model definitions remain the authors' original source. MedNet is cloned separately at the recorded commit. Our AST loader avoids training-only imports without modifying architecture class bodies. No trained MedNet checkpoint is bundled or claimed.

Local run from the workspace root:

```bash
.venv/bin/python -m src.pipeline --sample demo/imgs/2DBox_US_demo.npz --lite-repo upstream-LiteMedSAM --med-repo upstream-MedNet --checkpoint checkpoints/lite_medsam.pth --output outputs/integrated_local --device cpu
.venv/bin/python -m unittest discover -s tests -v
```

For your own image and human-provided box, call `src.pipeline.execute(image_rgb_uint8, box_xyxy, lite_repo, med_repo, checkpoint, output, device)` from a notebook. The demo CLI reads the authors' NPZ format. Output folders contain input/box, mask, overlays, crops, raw scores, source/checkpoint provenance, logs and package versions. `docs/LOCAL_RESULTS.json` preserves one measured run. Prepared GPU notebooks are not themselves evidence of GPU execution; see the task audit.

MedNet and the integrated pipeline have now also executed successfully on the Tesla T4 in appended cells of your existing Colab notebook. See docs/ASSIGNMENT_STATUS.md for exact evidence and remaining teaching/reproducibility work.
