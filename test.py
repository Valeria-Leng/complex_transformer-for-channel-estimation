import torch
import torch.nn as nn
import numpy as np
from scipy import io
from config import get_cfg
from torch.utils.data import DataLoader, TensorDataset
from model import Complex_transformer
from real import ChannelFromer
import os
from tqdm import tqdm
from model_training import huberloss
import matplotlib.pyplot as plt
cfg = get_cfg()

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class mseloss(nn.Module):
# class huberloss(nn.Module):
  def __init__(self,  device='cuda'):
    super(mseloss, self).__init__()
  def forward(self, true, pred):
    loss = torch.mean(torch.abs(pred - true)**2)
    return loss

def doppler_test():
      
    X_test_mat = io.loadmat('data/Doppler_X.mat')
    X_test = torch.tensor(X_test_mat['test_data'], dtype=torch.complex64) #torch.Size([11, 5000, 1, 72])
    Y_test_mat = io.loadmat('data/Doppler_Y.mat')
    Y_test = torch.tensor(Y_test_mat['test_y'], dtype=torch.complex64).squeeze() #torch.Size([11, 5000, (1,) 1008])
    if cfg.complex:  
        #model 
        #input = 256, 1, 72
        #output: 256, 1008
        model = Complex_transformer(input_dim=cfg.Nf, embed_dim=3*cfg.Nf, out_dim=cfg.Nf*cfg.Ns, fc_dim=cfg.Nf, num_heads=2).cuda()
    else:
        print('real!\n')
        X_test = torch.view_as_real(X_test).transpose(-1, -2)#torch.Size([11, 5000, 1,  2, 72])
        Y_test = torch.view_as_real(Y_test).transpose(-1, -2)
        model = ChannelFromer(input_dim=cfg.Nf, embed_dim=3*cfg.Nf, out_dim=cfg.Nf*cfg.Ns, num_heads=2).cuda()
        #input = 256, 1, 2, 72
        #output: 256, 2, 1008
    criterion = mseloss()

    # if cfg.test_flag:
    checkpoint = torch.load(cfg.path_checkpoint, map_location=torch.device(device))  # 加载断点
    model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
    # optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
    print('Load ckpt successfully!')
    for i in range(X_test.shape[0]):
        if cfg.complex:  
            X_t = X_test[i, :,:, :].to(device) #torch.Size([5000, 1, 72])
            Y_t = Y_test[i, :, :].to(device)
        else:
            X_t = X_test[i,:,:,:,:].to(device)
            Y_t = Y_test[i,:,:,:].to(device)
        model.eval()
        with torch.no_grad():
            tpred  = model(X_t)
            tloss = criterion(tpred, Y_t).item()
        if cfg.complex:    
            print('Doppler: {} Loss: {:.4f}'.format(i, tloss))
        else:
            print('Doppler: {} Loss: {:.4f}'.format(i, 2*tloss))
   


def loss_plot(Loss_list):
    x2 = range(cfg.epoch)
    y2 = Loss_list
    plt.plot(x2, y2, '.-')
    plt.xlabel('Training loss vs. epoches')
    plt.ylabel('Test loss')
    plt.show()
    if cfg.complex:
        plt.savefig("Complex_Training_loss.jpg")
    else:
        plt.savefig("Realbaseline_HA02_Training_loss.jpg")


if __name__ == '__main__':
    #
    complex_loss_list = np.load('complex_valid_loss.npy', allow_pickle=True)
    real_loss_list = np.load('real_valid_loss.npy', allow_pickle=True)

    x1 = range(100)
    x2 = range(50)
    plt.plot(x2, complex_loss_list, 'r', x1, real_loss_list, 'b')
    plt.xlabel('Validation loss vs. epoches')
    plt.ylabel('MSE')
    plt.show()
    plt.savefig("real vs cvnn loss.jpg")












    # doppler_test()
    #dataloader
    X_test_mat = io.loadmat('data/Test_X.mat')
    # print(X_train_mat['Training_X'].shape)
    X_test = torch.tensor(X_test_mat['Training_X'], dtype=torch.float32).permute(3, 2, 0, 1).contiguous()
    # X_test = torch.view_as_complex(X_test).unsqueeze(dim=1) #N, 1, 1, 72
    Y_test_mat = io.loadmat('data/Test_Y.mat')
    Y_test = torch.tensor(Y_test_mat['Training_Y'], dtype=torch.float32).permute(3, 2, 0, 1).contiguous()
    # Y_train = torch.view_as_complex(Y_train).view(Y_train.shape[0], cfg.Nf, cfg.Ns) #118750, 72, 14
    # Y_test = torch.view_as_complex(Y_test).squeeze() #118750, 1008
    
    if cfg.complex:
        print('complex!\n')
        X_test = torch.view_as_complex(X_test) #N, 1, 1, 72
        Y_test = torch.view_as_complex(Y_test).squeeze() #118750, 1008
        model = Complex_transformer(input_dim=cfg.Nf, embed_dim=3*cfg.Nf, out_dim=cfg.Nf*cfg.Ns, fc_dim=cfg.Nf, num_heads=2).cuda()
    else:
        print('real!\n')
        X_test = X_test.transpose(-1, -2) #torch.Size([35000, 1, 2, 72])
        Y_test = Y_test.transpose(-1, -2).squeeze()
        model = ChannelFromer(input_dim=cfg.Nf, embed_dim=3*cfg.Nf, out_dim=cfg.Nf*cfg.Ns, num_heads=2).cuda()
    #model 
    # model = Complex_transformer(input_dim =cfg.Nf, embed_dim = 3*cfg.Nf, out_dim=cfg.Nf*cfg.Ns, fc_dim=cfg.Nf, num_heads=2).to(device)
    criterion = mseloss()

    SNR = [0, 5, 10, 15, 20, 25, 30]
    # if cfg.test_flag:
    checkpoint = torch.load(cfg.path_checkpoint, map_location=torch.device(device))  # 加载断点
    model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
    # optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
    print('Load ckpt successfully!')
    for i in range(len(SNR)):
    
        X_t = X_test[5000*i:5000*(i+1),:,:].to(device)
        Y_t = Y_test[5000*i:5000*(i+1),:].to(device)


        model.eval()
        with torch.no_grad():
            tpred  = model(X_t) 
        if cfg.complex:
           tloss = criterion(tpred, Y_t).item()
           
        else:
           tloss = 2*criterion(tpred, Y_t).item()
           
        print('SNR: {} Loss: {:.4f}'.format(SNR[i], tloss))