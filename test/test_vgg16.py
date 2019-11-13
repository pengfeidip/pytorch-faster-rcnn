import torchvision as tv
import torch
import os.path as osp
import sys
cur_dir = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.join(cur_dir, '..'))

from lib import utils, config


if __name__ == '__main__':
    test_img = osp.join(cur_dir, 'dog.jpg')
    
    print('loading pretrained vgg16...')
    v = tv.models.vgg16(pretrained=True)
    v.to('cuda:0')
    v.eval()
    print('load image and pass through vgg16 network...')
    img_input = utils.imread(test_img).to('cuda:0')
    out = v(img_input)
    max_idx = out.argmax()
    print('argmax of output vector:', max_idx)
    print('image label:', utils.ImageNet_LABEL[max_idx.item()])

    top_modules = next(v.named_modules())
    print('Top level modules of VGG16:')
    print(top_modules)

    print('next try to utilize feature map of vgg')
    f = v.features
    a = v.avgpool
    c = v.classifier

    print('output by pass through the whole network:')
    print(out[0][:10], ' ...')
    out2 = f(img_input)
    out2 = a(out2)
    out2 = out2.view(1, -1)
    out2 = c(out2)

    print('output by pass through each component of the network:')
    print(out2[0][:10], ' ...')

    for p in v.parameters():
        print(p.requires_grad)

    
