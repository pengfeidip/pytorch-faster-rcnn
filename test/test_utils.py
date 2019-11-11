import os.path as osp
import os, sys

cur_dir = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.join(cur_dir, '..'))

from lib import data, modules, utils, region

def test_split_list():
    a = list(range(10))
    for i in utils.split_list(a, 2):
        print(i)

if __name__ == '__main__':
    test_split_list()

