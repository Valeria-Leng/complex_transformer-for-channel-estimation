from yacs.config import CfgNode as CN
import numpy as np
import math

_CN = CN()
_CN.Nf=72
_CN.Ns=14
_CN.batch_size = 128
_CN.epoch=100
_CN.filename = './Light_ChannelFromer/theory'
_CN.path_checkpoint = _CN.filename+'/ckpt_best_ever39.pth'  # 断点路径
# _CN.pkl_path = 'training_data/data.pkl'

_CN.test_flag = False
_CN.doppler = False
_CN.complex = False
_CN.no_CLN = False
_CN.no_Cattention = True
def get_cfg():
    return _CN.clone()