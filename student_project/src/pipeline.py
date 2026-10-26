"""Demonstration-only segmentation -> crop -> untrained two-score classification."""
import argparse, hashlib, json, platform, subprocess, time
from pathlib import Path
import cv2
import numpy as np
import torch
from .segmentation import segment
from .cropping import crop_regions
from .classification import build_mednet,preprocess_crop,classify

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def revision(repo): return subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
def execute(image,box,lite_repo,med_repo,checkpoint,output,device='cpu'):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    start=time.perf_counter()
    mask,seg_report=segment(image,box,lite_repo,checkpoint,output,device)
    cv2.imwrite(str(output/'image.png'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output/'mask.png'),mask*255)
    (output/'box.json').write_text(json.dumps(np.asarray(box).tolist()))
    model,model_report=build_mednet(Path(med_repo)/'MedNet.py',device)
    crop_reports=[];crop_dir=output/'crops';crop_dir.mkdir(exist_ok=True)
    for item in crop_regions(image,mask):
        name=f"crop_margin_{round(item['margin']*100):02d}.png"
        cv2.imwrite(str(crop_dir/name),cv2.cvtColor(item['image'],cv2.COLOR_RGB2BGR))
        tensor=preprocess_crop(item['image'])
        scores=classify(model,tensor)
        crop_reports.append({k:v for k,v in item.items() if k!='image'} | {'file':name,'input_shape':list(tensor.shape)} | scores)
    report={'segmentation':seg_report,'classifier':model_report,'crops':crop_reports,
            'source_commits':{'LiteMedSAM':revision(lite_repo),'MedNet':revision(med_repo)},
            'checkpoint_sha256':sha(checkpoint),'python':platform.python_version(),'torch':torch.__version__,
            'total_seconds':time.perf_counter()-start,
            'preprocessing':'RGB->BGR; resize 224x224 linear; divide 255; mean [.485,.456,.406], std [.229,.224,.225]',
            'status':'Executed software prototype; MedNet is untrained; no clinical interpretation.'}
    (output/'scores_and_run.json').write_text(json.dumps(report,indent=2))
    (output/'requirements-run.txt').write_text(subprocess.check_output([__import__('sys').executable,'-m','pip','freeze'],text=True))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sample',required=True)
    p.add_argument('--lite-repo',required=True);p.add_argument('--med-repo',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu');a=p.parse_args()
    with np.load(a.sample,allow_pickle=False) as data: image,box=data['imgs'],data['boxes'][0]
    try: print(json.dumps(execute(image,box,a.lite_repo,a.med_repo,a.checkpoint,a.output,a.device),indent=2))
    except Exception as e:
        Path(a.output).mkdir(parents=True,exist_ok=True)
        (Path(a.output)/'failure.json').write_text(json.dumps({'error':type(e).__name__,'message':str(e)}))
        raise
