from . import utils

def tensor_shape(tsr):
    if isinstance(tsr, list):
        for x in tsr:
            print('  ' + str(x.shape))
    else:
        print(tsr.shape)

def count_tensor(tsr):
    print(utils.count_tensor(tsr))
