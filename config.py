from yacs.config import CfgNode as CN
import numpy as np
import math

_CN = CN()
_CN.Nf=72
_CN.Ns=14
_CN.batch_size = 128
_CN.epoch=50
_CN.filename = './real_baseline'
_CN.path_checkpoint = _CN.filename+'/ckpt_best_19.pth'  # 断点路径
# _CN.pkl_path = 'training_data/data.pkl'

_CN.test_flag = False
_CN.doppler = False
_CN.complex = True
def get_cfg():
    return _CN.clone()