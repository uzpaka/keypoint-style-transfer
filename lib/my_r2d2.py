# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import os, pdb
from PIL import Image
import numpy as np
import torch
import cv2

#from tools import common
from lib.my_r2d2_patchnet import *

import torchvision.transforms as tvf
RGB_mean = [0.485, 0.456, 0.406]
RGB_std  = [0.229, 0.224, 0.225]
norm_RGB = tvf.Compose([tvf.ToTensor(), tvf.Normalize(mean=RGB_mean, std=RGB_std)])

class NonMaxSuppression (torch.nn.Module):
    def __init__(self, rel_thr=0.7, rep_thr=0.7):
        nn.Module.__init__(self)
        self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.rel_thr = rel_thr
        self.rep_thr = rep_thr
    
    def forward(self, reliability, repeatability, **kw):
        assert len(reliability) == len(repeatability) == 1
        reliability, repeatability = reliability[0], repeatability[0]

        # local maxima
        maxima = (repeatability == self.max_filter(repeatability))

        # remove low peaks
        maxima *= (repeatability >= self.rep_thr)
        maxima *= (reliability   >= self.rel_thr)

        #return maxima.nonzero().t()[2:4]
        return maxima.nonzero(as_tuple=False).t()[2:4]
    

def load_network(model_fn): 
    checkpoint = torch.load(model_fn)
    #print("\n>> Creating net = " + checkpoint['net']) 
    net = eval(checkpoint['net'])

    # initialization
    weights = checkpoint['state_dict']
    net.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})
    return net.eval()



def extract_multiscale(net, img):
                     
   # old_bm = torch.backends.cudnn.benchmark 
   # torch.backends.cudnn.benchmark = False # speedup
    
    B, three, H, W = img.shape
    assert B == 1 and three == 3, "should be a batch with a single RGB image"
        
    with torch.no_grad():
        res = net(imgs=[img])
                       
    descriptors = res['descriptors'][0]
    #reliability = res['reliability'][0]
    #repeatability = res['repeatability'][0]

    # restore value
    #torch.backends.cudnn.benchmark = old_bm

    return descriptors


def r2d2_extract_sparse_descriptors(img_rgb_numpy=None, keypoints=None, net = load_network("models/r2d2_WASF_N16.pt")):
    #keypoints: [N,2] where N=nb keypoints and stored as x,y (width, height)
                       
    #iscuda = common.torch_set_gpu(args.gpu)
    iscuda = True
    
    if iscuda: net = net.cuda()

    #img = Image.fromarray(img_rgb_numpy, 'RGB')
   #W, H = img.size
    img = norm_RGB(img_rgb_numpy)[None] 
    if iscuda: img = img.cuda()
                       
    # extract descriptors for the image
    desc = extract_multiscale(net, img)   #[1,128,h,w]
    sparse_descriptors = desc[0,:,keypoints[:,1], keypoints[:,0]].t()  #Nx128
    
    #print(sparse_descriptors.shape)
    #print("#######################")
    #print(sparse_descriptors[1,:])
    #print("#########################")
    #print(desc[0,:,2,0])
    
    return sparse_descriptors, net


def r2d2_extract_keypoints_and_sparse_descriptors(img_rgb_numpy=None, net = load_network("models/r2d2_WASF_N16.pt"), max_nb_keypoints=5000):
                       
    #iscuda = common.torch_set_gpu(args.gpu)
    iscuda = True
    
    if iscuda: net = net.cuda()

    #img = Image.fromarray(img_rgb_numpy, 'RGB')
    img = norm_RGB(img_rgb_numpy)[None] 
    if iscuda: img = img.cuda()
                       
    with torch.no_grad():
        res = net(imgs=[img])
                       
    descriptors = res['descriptors'][0]
    reliability = res['reliability'][0]
    repeatability = res['repeatability'][0]
    
    detector = NonMaxSuppression(rel_thr = 0.7, rep_thr = 0.7)
    y,x = detector(**res) # nms
    c = reliability[0,0,y,x]
    q = repeatability[0,0,y,x]
    d = descriptors[0,:,y,x].t()
    #n = d.shape[0]
    
    X,Y,C,Q,D = [],[],[],[],[]
    
    X.append(x)
    Y.append(y)
    #S.append((32/1.0) * torch.ones(n, dtype=torch.float32, device=d.device))
    C.append(c)
    Q.append(q)
    D.append(d)
    
    Y = torch.cat(Y)
    X = torch.cat(X)
    #S = torch.cat(S) # scale
    scores = torch.cat(C) * torch.cat(Q) # scores = reliability * repeatability
    #XYS = torch.stack([X,Y,S], dim=-1)
    XY = torch.stack([X,Y], dim=-1)
    D = torch.cat(D)
    
    desc = D
    keypoints = XY.cpu().numpy()
    scores = scores.cpu().numpy()
    idxs = scores.argsort()[-max_nb_keypoints or None:] #Take the best 5000 keypoints...
    
    keypoints = keypoints[idxs] 
    sparse_descriptors = desc[idxs]
    scores = scores[idxs]
    
#    print(keypoints.shape)           #[N, 2]
#    print(sparse_descriptors.shape)  #[N,128]
#    print(scores.shape)              #[N,]
    
    return sparse_descriptors, keypoints, net


def r2d2_dense_output(input_img, net, input_ready):
                       
    iscuda = True
    
    if not input_ready:
        #input is a numpy array, RGB in values [0,255] of shape [H,W,3]
        img = norm_RGB(input_img)[None] 
        img = img.cuda()
    else:
        #input is a torch tensor image RGB in values [0,1], mean centered and divided by std, of shape [1,C,H,W]
        img = input_img
        
    #if iscuda: img = img.cuda()
                       
    #with torch.no_grad():
    res = net(imgs=[img])
                       
    dense_descriptors = res['descriptors'][0]  #(1,128,h,w)
    reliability = res['reliability'][0]        #(1,1,h,w)
    repeatability = res['repeatability'][0]    #(1,1,h,w)
    
    #y,x = detector(**res) # nms
    #c = reliability[0,0,y,x]
    #q = repeatability[0,0,y,x]

    scores = reliability[0,:,:,:] * repeatability[0,:,:,:]
    
    return dense_descriptors, scores

if __name__ == '__main__':

    #r2d2_extract_sparse_descriptors(img_rgb_numpy=cv2.cvtColor(cv2.imread("/media/uzpaka/hdd1/datasets/DNIM/Image/00023966/20151120_054642.jpg"), cv2.COLOR_BGR2RGB), keypoints = np.array([[0,1], [0,2], [5,5]]))

    r2d2_extract_keypoints_and_sparse_descriptors(img_rgb_numpy=cv2.cvtColor(cv2.imread("/media/uzpaka/hdd1/datasets/DNIM/Image/00023966/20151120_054642.jpg"), cv2.COLOR_BGR2RGB))
