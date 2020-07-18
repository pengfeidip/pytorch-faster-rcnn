import torch
import torchvision as tv
from mmdet.models import build_backbone
from lib.backbones import ResNet

BB_CFG = {
    50: dict(type='ResNet', depth=50),
    101: dict(type='ResNet', depth=101)
}

RES_NET = {
    50 : tv.models.resnet50,
    101: tv.models.resnet101
}

CKPT = {
    50 : '/home/lee/.cache/torch/checkpoints/resnet50-19c8e357.pth',
    101: '/home/lee/.cache/torch/checkpoints/resnet101-5d3b4d8f.pth'
}

def compare_res(n=50):
    print('Compare resnet:', n)
    assert n in RES_NET
    tres = RES_NET[n]()
    lres = ResNet(n)

    pre = torch.load(CKPT[n])
    tres.load_state_dict(pre)
    lres.load_state_dict(pre)

    img_data = torch.rand(10, 3, 224, 224)
    tout = tres(img_data)
    lout = lres(img_data)

    diff = (tout - lout).sum().item()
    print('diff:', diff)

def load_dict(net, wt):
    try:
        return net.load_state_dict(wt)
    except:
        pass

def compare_backbone(n=50):
    assert n in BB_CFG
    print('Compare backbone:', n)
    # backbone from mmdet
    mmdet_res = build_backbone(BB_CFG[n])
    lyq_res = ResNet(n)
    pre = torch.load(CKPT[n])

    load_dict(mmdet_res, pre)
    load_dict(lyq_res, pre)

    mmdet_res.train()
    lyq_res.train()


    img_data = torch.rand(10, 3, 224, 224)
    mmdet_feats = mmdet_res(img_data)
    lyq_feats = lyq_res(img_data)

    print('Number of feats from mmdet backbone:', len(mmdet_feats))
    print('Number of feats from lyq backbone:', len(lyq_feats))

    for i in range(len(mmdet_feats)):
        mf = mmdet_feats[i]
        lf = lyq_feats[i]
        diff = (mf-lf).sum().item()
        print('diff:', diff)
        


if __name__ == '__main__':
    compare_backbone(50)
    compare_backbone(101)
    pass
