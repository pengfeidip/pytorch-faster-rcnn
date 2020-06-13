from . import utils
import torch

def tensor_shape(tsr, words=None):
    if words is not None:
        print(words)
    print(utils.tensor_shape(tsr))

def count_tensor(tsr, words=None):
    if words is not None:
        print(words)
    print(utils.count_tensor(tsr))

def peek_tensor(tsr):
    assert isinstance(tsr, torch.Tensor)
    out_str = 'size:{}, dtype:{}, device:{}'\
              .format(tsr.shape, tsr.dtype, tsr.device)
    print(out_str)

def vis_2d(tsr, trans=lambda x:x):
    row, col = tsr.shape[:2]
    tab = []
    for i in range(row):
        rows = []
        for j in range(col):
            rows.append(str(trans(tsr[i, j])))
        tab.append(rows)
    lines = [' '.join(x) for x in tab]
    tab_str = '\n'.join(lines)
    print(tab_str)
    
