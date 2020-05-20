import torch

def map2level(scale, strides, sides):
    min_scale = scale * strides[0]
    sides = sides / min_scale
    sides = torch.log2(sides).floor().long().clamp(0, len(strides)-1)
    return sides
    

def test():
    scale = 12
    strides = torch.tensor([4, 8, 16, 32, 64])
    sides = torch.tensor([20, 48, 66, 127, 1000, 200000]).float()
    print('scales:', [scale * x for x in strides])
    print('sides:', sides)
    print('map loc:', map2level(scale, strides, sides))
    

if __name__ == '__main__':
    test()
