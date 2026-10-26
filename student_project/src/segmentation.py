"""Execute the unmodified official LiteMedSAM CLI with a supplied human box."""
import json, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch

def segment(image, box, repo, checkpoint, output, device='cpu'):
    repo,checkpoint,output=Path(repo).resolve(),Path(checkpoint).resolve(),Path(output).resolve()
    if image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8:
        raise ValueError('Expected uint8 RGB HWC image.')
    h,w=image.shape[:2]; box=np.asarray(box,dtype=np.float32)
    if box.shape!=(4,) or not np.isfinite(box).all() or not (0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h):
        raise ValueError('Expected a finite, nonempty xyxy box within image bounds.')
    if device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA GPU unavailable.')
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    inputs=output/'inputs';preds=output/'masks';overlays=output/'overlays'
    inputs.mkdir(parents=True,exist_ok=True);preds.mkdir(exist_ok=True)
    np.savez_compressed(inputs/'2DBox_demo.npz',imgs=image,boxes=box[None,:])
    command=[sys.executable,'CVPR24_LiteMedSAM_infer.py','-i',str(inputs),'-o',str(preds),
             '-lite_medsam_checkpoint_path',str(checkpoint),'-device',device,'-png_save_dir',str(overlays)]
    start=time.perf_counter()
    run=subprocess.run(command,cwd=repo,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (output/'segmentation.log').write_text(run.stdout)
    run.check_returncode()
    with np.load(preds/'2DBox_demo.npz',allow_pickle=False) as p: mask=p['segs'].copy()
    if mask.shape!=(h,w) or not set(np.unique(mask)).issubset({0,1}): raise ValueError('Invalid output mask.')
    return mask,{'device':device,'command':command,'wall_seconds_includes_subprocess_loading_io':time.perf_counter()-start}
