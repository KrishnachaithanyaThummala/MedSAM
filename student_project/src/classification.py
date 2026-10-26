"""Import the exact upstream architecture without training-only imports or side effects."""
import ast
import hashlib
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn

MODEL_CLASSES = ('ChannelAttention','SpatialAttention','CBAM','DepthwiseSeparableConv',
                 'ResidualDSCBAMBlock','ResidualCBAMBackbone','Darma')

def build_mednet(source, device='cpu', seed=42):
    source = Path(source)
    tree = ast.parse(source.read_text())
    selected = [node for node in tree.body if isinstance(node,ast.ClassDef) and node.name in MODEL_CLASSES]
    if tuple(node.name for node in selected) != MODEL_CLASSES:
        raise ValueError('Upstream architecture class list changed; inspect before running.')
    namespace = {'torch':torch,'nn':nn}
    # Compile original AST class bodies unchanged. Omit only unrelated imports/training utilities.
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),namespace)
    # Restore the caller's RNG state after reproducible untrained initialization.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = namespace['Darma'](num_classes=2,use_cbam=True)
    model = model.to(device).eval()
    provenance = {'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                  'class':'Darma','weights':'random initialization; no trained checkpoint loaded',
                  'seed':seed,'num_classes':2,'architecture_classes':list(MODEL_CLASSES)}
    return model,provenance

def preprocess_crop(rgb, size=224):
    if rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8 or rgb.size==0:
        raise ValueError('Expected nonempty uint8 RGB crop.')
    # Match MedNet.py's AlbumentationsTransform: RGB->BGR, resize, ImageNet constants.
    bgr = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    resized = cv2.resize(bgr,(size,size),interpolation=cv2.INTER_LINEAR).astype(np.float32)/255.
    mean = np.array([.485,.456,.406],dtype=np.float32)
    std = np.array([.229,.224,.225],dtype=np.float32)
    return torch.from_numpy(((resized-mean)/std).transpose(2,0,1).copy()).unsqueeze(0)

def classify(model, tensor):
    device = next(model.parameters()).device
    with torch.inference_mode():
        scores,activations = model(tensor.to(device))
    if scores.shape != (tensor.shape[0],2) or not torch.isfinite(scores).all():
        raise ValueError('Expected finite batch x 2 logits.')
    return {'raw_scores':scores.cpu().tolist(),'output_shape':list(scores.shape),
            'activation_shapes':[list(x.shape) for x in activations],
            'interpretation':'Untrained numerical scores only; class meanings are not assigned.'}
