import torch
import torch.nn as nn
import torch.optim as optim
from scipy import io
import numpy as np
from config import get_cfg
from torch.utils.data import DataLoader, TensorDataset
from model import Complex_transformer
# from model2 import Complex_transformer
from real import ChannelFromer
import os
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
cfg = get_cfg()
import matplotlib.pyplot as plt
# from test import mseloss
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class mseloss(nn.Module):
# class huberloss(nn.Module):
  def __init__(self,  device='cuda'):
    super(mseloss, self).__init__()
  def forward(self, true, pred):
    loss = torch.mean(torch.abs(pred - true)**2)
    return loss

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



# huber 损失
class huberloss(nn.Module):
    def __init__(self, delta=1.0, device='cuda'):
        super(huberloss, self).__init__()
        self.delta = delta
    def forward(self, true, pred):
        abs = torch.abs(true-pred)
        loss = torch.where(abs < self.delta , 0.5*(abs**2), self.delta*abs - 0.5*(self.delta**2))
        return torch.mean(loss)

if __name__ == '__main__':
    #dataloader
    X_train_mat = io.loadmat('data/Training_X.mat')
    # print(X_train_mat['Training_X'].shape)
    X_train = torch.tensor(X_train_mat['Training_X'], dtype=torch.float32).permute(3, 2, 0, 1).contiguous() #torch.Size([118750, 1, 72, 2])
    
    Y_train_mat = io.loadmat('data/Training_Y.mat')
    Y_train = torch.tensor(Y_train_mat['Training_Y'], dtype=torch.float32).permute(3, 2, 0, 1).contiguous() #torch.Size([118750, 1, 1008, 2])
    # Y_train = torch.view_as_complex(Y_train).view(Y_train.shape[0], cfg.Nf, cfg.Ns) #118750, 72, 14

    X_val_mat = io.loadmat('data/Validation_X.mat')
    # print(X_train_mat['Validation_X'].shape)
    X_val = torch.tensor(X_val_mat['Validation_X'], dtype=torch.float32).permute(3, 2, 0, 1).contiguous()
    
    Y_val_mat = io.loadmat('data/Validation_Y.mat')
    Y_val = torch.tensor(Y_val_mat['Validation_Y'], dtype=torch.float32).permute(3, 2, 0, 1).contiguous()
    # Y_val = torch.view_as_complex(Y_val).view(Y_val.shape[0], cfg.Nf, cfg.Ns) #N, 72, 14
    
    if cfg.complex:
        X_train = torch.view_as_complex(X_train) #N, 1, 1, 72
        Y_train = torch.view_as_complex(Y_train).squeeze() #118750, 1008
        X_val = torch.view_as_complex(X_val) #N, 1, 1, 72
        Y_val = torch.view_as_complex(Y_val).squeeze() #N, 72, 14
        model = Complex_transformer(input_dim=cfg.Nf, embed_dim=3*cfg.Nf, out_dim=cfg.Nf*cfg.Ns, fc_dim=cfg.Nf, num_heads=2).cuda()
        print("Complex HA02 is training!")
    else:
        X_train = X_train.transpose(-1, -2)
        X_val = X_val.transpose(-1, -2)
        Y_train = Y_train.transpose(-1, -2).squeeze()
        Y_val = Y_val.transpose(-1, -2).squeeze()
        model = ChannelFromer(input_dim=cfg.Nf, embed_dim=3*cfg.Nf, out_dim=cfg.Nf*cfg.Ns, num_heads=2).cuda()
        print("Real baseline HA02 is training!")
        


    TrainDataset = TensorDataset(X_train, Y_train)
    train_dl = DataLoader(TrainDataset, batch_size=cfg.batch_size, shuffle=True)
    # ValDataset = TensorDataset(X_val, Y_val)
    # val_dl = DataLoader(ValDataset, batch_size=cfg.batch_size, shuffle=True)
    #model
    # model = Complex_transformer(input_dim =cfg.Nf, embed_dim = 3*cfg.Nf, out_dim=cfg.Nf*cfg.Ns, fc_dim=cfg.Nf, num_heads=2).cuda()
    criterion = huberloss()
    test_criterion = mseloss()
    # optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-7)
    # lr_schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.5)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-7)
    lr_schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 12], gamma=0.5)

    if cfg.test_flag:
        checkpoint = torch.load(cfg.path_checkpoint, map_location=torch.device(device))  # 加载断点
        model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
        lr_schedule.load_state_dict(checkpoint['lr_schedule'])
        #  lr_schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90,97],gamma=0.1,last_epoch=0)
        print('Load epoch {} successfully'.format(start_epoch))

    else:
        start_epoch = -1
        print('From epoch 0')
        print("------------------------------Ready for training!-------------------------")
        # print('L:{}\t M:{}\t N:{}\t K:{}\t Complex?:{}\t'.format(cfg.L, cfg.M, cfg.N, cfg.K, cfg.complex))
    train_loss = []
    valid_loss = []
    for epoch in range(start_epoch+1, cfg.epoch):
        model.train()
        loss_sum = 0
        for index, (x_train, y_train) in enumerate(tqdm(train_dl)):

            x_train = x_train.to(device)
            y_train = y_train.to(device)
            out = model(x_train)
            if cfg.complex:
                loss = criterion(out, y_train)
            else:
                loss = 2*criterion(out, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        train_loss.append(loss.item())
        loss_mean = loss_sum / len(train_dl)
        print('Train Epoch: {}\t Loss: {:.6f}'.format(epoch, loss_mean))
        

        model.eval()
        with torch.no_grad():
            X_val = X_val.to(device)
            Y_val = Y_val.to(device)
            pred = model(X_val)
            if cfg.complex:
                vloss = test_criterion(pred, Y_val).item()
            else:
                vloss = 2*test_criterion(pred, Y_val).item()
        valid_loss.append(vloss)
        print('Val set: Loss: {:.4f}'.format(vloss))
        print('learning rate:',optimizer.state_dict()['param_groups'][0]['lr'])
        print('\n')

        lr_schedule.step()

        # checkpoint = {
        #     'net': model.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        #     'epoch': epoch,
        #     'lr_schedule': lr_schedule.state_dict()
        # }
        # if not os.path.isdir(cfg.filename):
        #     os.mkdir(cfg.filename)
        # torch.save(checkpoint, cfg.filename+'/ckpt_best_%s.pth' % (str(epoch)))
    if cfg.complex:

        np.save('complex_train_loss.npy',train_loss) 
        np.save('complex_valid_loss.npy',valid_loss) 
    else:
        np.save('real_train_loss.npy',train_loss) 
        np.save('real_valid_loss.npy',valid_loss)

    loss_plot(train_loss)