import torch
import time
import yaml
from src import utils
from torch import nn
from easydict import EasyDict
from src.DuAT.DuAT import DuAT
from src.Unet.Unet import UNet
from src.Unetr.Unetr import u_netr
from datetime import datetime
from src.SwinUNETR.SwinUNETR import swin_unetr
from src.CFPnet.CFPnet import CFPNet
from src.TransUnet.TransUnet import TransUNet
from src.CVCUNETR.CVCUNETR import CVCUnetr
from src.CVCUNETR.NewCVC import CVC_Unetr
from src.CVCUNETR.CVCUNETRv4 import CVC_UnetrV4
import warnings
warnings.filterwarnings('ignore')


def test_weight(model, x):
    # torch.cuda.synchronize()
    start_time = time.time()
    _ = model(x)
    # torch.cuda.synchronize()
    end_time = time.time()
    # torch.cuda.synchronize()
    need_time = end_time - start_time
    from thop import profile
    flops, params = profile(model, inputs=(x,))
    throughout = round(x.shape[0] / (need_time / 1), 3)
    return flops, params, throughout


def Unitconversion(name, flops, params, throughout):
    print('Model name : {}.'.format(name))
    print('params : {} M'.format(round(params / 10000000, 2)))
    print('flop : {} G'.format(round(flops / 10000000000, 2)))
    print('throughout: {} FPS'.format(throughout))
    print("-"*100)

def get_result(model_name, model, x):
    for i in range(0, 10):
        _ = model(x)
    flops, param, throughout = test_weight(model, x)
    Unitconversion(model_name, flops, param, throughout)

if __name__ == '__main__':
    # 读取配置
    device = 'cuda:0'
    config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    utils.same_seeds(50)
    
    x = torch.randn(size=(1, 3, 352, 352)).to(device=device)

    model = CVCUnetr(**config.models.cvc_unetr.branch1).to(device=device)
    get_result('CVCUnetr',model,x)
    time.sleep(1)
    model = CVC_Unetr(**config.models.cvc_unetr.branch1).to(device=device)
    get_result('CVC_Unetr',model,x)
    time.sleep(1)
    model = CVC_UnetrV4(**config.models.cvc_unetr_v4.branch1).to(device=device)
    get_result('CVC_UnetrV4', model, x)
    time.sleep(1)
    model = TransUNet(**config.models.trans_unet.branch1).to(device=device)
    get_result('TransUNet',model,x)
    time.sleep(1)
    model = CFPNet(**config.models.cfp_net.branch1).to(device=device)
    get_result('CFPNet',model,x)
    time.sleep(1)
    model = u_netr(**config.models.u_netr.branch1).to(device=device)
    get_result('u_netr',model,x)
    time.sleep(1)
    model = swin_unetr(**config.models.swin_unetr.branch1).to(device=device)
    get_result('swin_unetr',model,x)
    time.sleep(1)
    model = DuAT(**config.models.duat.branch1).to(device=device)
    get_result('DuAT',model,x)
    time.sleep(1)
    model = UNet(**config.models.unet.branch1).to(device=device)
    get_result('u_net',model,x)

    
