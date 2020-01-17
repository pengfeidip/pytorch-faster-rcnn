import os, sys, torch
import os.path as osp
lib_path = osp.join(osp.dirname(osp.realpath(__file__)), '..')
sys.path.insert(0, lib_path)

from lib import faster_rcnn

def weight_norm(w):
    num_w = w.shape[0]
    w = w.view(num_w, -1)
    wnorm = torch.norm(w, dim=1)
    return wnorm.sum()/num_w

def bias_norm(b):
    num_b = b.shape[0]
    bnorm = torch.norm(b)
    return bnorm

if __name__ == '__main__':
    try:
        pth = sys.argv[1]
    except:
        print('\nUsage:', __file__, 'ckpt\n')
        exit()
    faster = faster_rcnn.FasterRCNNModule()
    faster.load_state_dict(torch.load(pth))

    weights, bias = faster.conv_parameters()
    for w in weights:
        print(weight_norm(w).tolist())
    for b in bias:
        print(bias_norm(b).tolist())
