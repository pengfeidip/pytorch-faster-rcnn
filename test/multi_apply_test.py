import sys, os, time
import os.path as osp
cur_dir = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.join(cur_dir, '..'))
from lib import utils

def func(a, b, c):
    return a, b, c

def func_one(a, b, c):
    return a

def test():
    a = 'a'
    b = ['b1', 'b2']
    c = 'c'
    result = utils.multi_apply(func, a, b, c)
    print('result of multi_apply')
    print(result)

    unpack_res = utils.unpack_multi_result(result)
    print('unpacked result')
    print(unpack_res)
    


if __name__ == '__main__':
    test()
