"""Mask-derived crops. Each margin expands each side by a fraction of tight width/height."""
import math
import numpy as np

def crop_regions(image, mask, margins=(0.,.1,.2)):
    if image.ndim!=3 or image.shape[2]!=3 or mask.shape!=image.shape[:2]:
        raise ValueError('Expected RGB image and a matching 2D mask.')
    ys,xs = np.where(mask>0)
    if not len(xs): raise ValueError('Empty segmentation: no crop can be defined.')
    h,w=mask.shape
    x0,y0,x1,y1=int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1
    result=[]
    for margin in margins:
        if not np.isfinite(margin) or margin<0: raise ValueError('Margins must be finite and nonnegative.')
        dx,dy=math.ceil((x1-x0)*margin),math.ceil((y1-y0)*margin)
        bounds=[max(0,x0-dx),max(0,y0-dy),min(w,x1+dx),min(h,y1+dy)]
        a,b,c,d=bounds
        result.append({'margin':float(margin),'box_xyxy_exclusive':bounds,'image':image[b:d,a:c].copy()})
    return result
