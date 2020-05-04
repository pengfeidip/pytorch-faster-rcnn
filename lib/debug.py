
def tensor_shape(tsr):
    if isinstance(tsr, list):
        for x in tsr:
            print('  ' + str(x.shape))
    else:
        print(tsr.shape)
