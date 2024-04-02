import torch
import torch.nn as nn
import torch.optim as optim
from scipy import io
from config import get_cfg
from torch.utils.data import DataLoader, TensorDataset
# from model import Complex_transformer
from complex_light import Complex_transformer
from real_light import Light_ChannelFromer
import os
from tqdm import tqdm
import numpy as np
from test import mseloss
cfg = get_cfg()

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# huber 损失
class huberloss(nn.Module):
    def __init__(self, delta=1.0, device='cuda'):
        super(huberloss, self).__init__()
        self.delta = delta
    def forward(self, true, pred):
        abs = torch.abs(true-pred)
        loss = torch.where(abs < self.delta , 0.5*(abs**2), self.delta*abs - 0.5*(self.delta**2))
        return torch.mean(loss)


def test():
    if cfg.doppler:
        X_test_mat = io.loadmat('data/new/doppler/test_30X.mat')
        X_test = torch.tensor(X_test_mat['test_data'], dtype=torch.complex64).transpose(-1,-2) #(11, 5000, 36, 2)
        Y_test_mat = io.loadmat('data/new/doppler/test_30Y.mat')
        Y_test = torch.tensor(Y_test_mat['test_y'], dtype=torch.complex64).transpose(-1,-2) 
        print('SNR = 30db')
    else:
        #dataloader
        X_test_mat = io.loadmat('data/new/new_Test_X.mat')
        # print(X_train_mat['Training_X'].shape)
        X_test = torch.tensor(X_test_mat['Training_X'], dtype=torch.complex64).permute(2, 1, 0).contiguous()
        Y_test_mat = io.loadmat('data/new/new_Test_Y.mat')
        Y_test = torch.tensor(Y_test_mat['Training_Y'], dtype=torch.complex64).permute(2, 1, 0).contiguous()

    #model 
    print('complex?',cfg.complex)
    if cfg.complex:
        model = Complex_transformer(input_dim =X_test.shape[-1], out_dim=Y_test.shape[-2], fc_dim=Y_test.shape[-1], num_heads=2).cuda()
    else:
        model = Light_ChannelFromer(input_dim =X_test.shape[-1], out_dim=Y_test.shape[-1], num_heads=2).cuda()
    # model = Complex_transformer(input_dim =X_test.shape[-1], out_dim=Y_test.shape[-2], fc_dim=Y_test.shape[-1], num_heads=2).cuda()
    #input = 256, 2, 36
    #output: 256, 14, 72
    criterion = mseloss()

    # if cfg.test_flag:
    checkpoint = torch.load(cfg.path_checkpoint, map_location=torch.device(device))  # 加载断点
    model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
    # optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
    print('Load ckpt successfully!')

    if cfg.doppler:
        for i in range(X_test.shape[0]):
            X_t = X_test[i, :,:, :].to(device)
            Y_t = Y_test[i, :, :, :].to(device)
            model.eval()
            with torch.no_grad():
                tpred  = model(X_t)
                tloss = criterion(tpred, Y_t).item()
            
            print('Doppler: {} Loss: {:.4f}'.format(i, tloss))

    else:
        SNR = [0, 5, 10, 15, 20, 25, 30]
        for i in range(len(SNR)):
        
            X_t = X_test[5000*i:5000*(i+1),:,:].to(device)
            Y_t = Y_test[5000*i:5000*(i+1),:,:].to(device)

            model.eval()
            with torch.no_grad():
                tpred  = model(X_t)
                tloss = criterion(tpred, Y_t).item()
            
            print('SNR: {} MSE: {:.4f}'.format(SNR[i], tloss))


if __name__ == '__main__':

    test()
    #dataloader
    X_train_mat = io.loadmat('data/new/new_Training_X.mat')
    # print(X_train_mat['Training_X'].shape)
    X_train = torch.tensor(X_train_mat['Training_X'], dtype=torch.complex64).permute(2, 1, 0).contiguous() #torch.Size([36, 2, 118750])
    # X_train = torch.view_as_complex(X_train).unsqueeze(dim=1) #N, 1, 1, 72
    Y_train_mat = io.loadmat('data/new/new_Training_Y.mat')
    Y_train = torch.tensor(Y_train_mat['Training_Y'], dtype=torch.complex64).permute(2, 1, 0).contiguous() #torch.Size([72, 14, 118750])
    


    X_val_mat = io.loadmat('data/new/new_Validation_X.mat')
    # print(X_train_mat['Validation_X'].shape)
    X_val = torch.tensor(X_val_mat['Validation_X'], dtype=torch.complex64).permute(2, 1, 0).contiguous()
    # X_val = torch.view_as_complex(X_val).unsqueeze(dim=1) #N, 1, 1, 72
    Y_val_mat = io.loadmat('data/new/new_Validation_Y.mat')
    Y_val = torch.tensor(Y_val_mat['Validation_Y'], dtype=torch.complex64).permute(2, 1, 0).contiguous()
    
    X_test_mat = io.loadmat('data/new/new_Test_X.mat')
    # print(X_train_mat['Training_X'].shape)
    X_test = torch.tensor(X_test_mat['Training_X'], dtype=torch.complex64).permute(2, 1, 0).contiguous()
    Y_test_mat = io.loadmat('data/new/new_Test_Y.mat')
    Y_test = torch.tensor(Y_test_mat['Training_Y'], dtype=torch.complex64).permute(2, 1, 0).contiguous()



    TrainDataset = TensorDataset(X_train, Y_train)
    train_dl = DataLoader(TrainDataset, batch_size=cfg.batch_size, shuffle=True)
    # ValDataset = TensorDataset(X_val, Y_val)
    # val_dl = DataLoader(ValDataset, batch_size=cfg.batch_size, shuffle=True)
    #model
    model = Complex_transformer(input_dim =X_train.shape[-1], out_dim=Y_train.shape[-2], fc_dim=Y_train.shape[-1], num_heads=2).cuda() 
    # model = Light_ChannelFromer(input_dim =X_train.shape[-1], out_dim=Y_train.shape[-1], num_heads=2).cuda()    

    #input = 256, 2, 36
    #output: 256, 14, 72
    criterion = huberloss()
    loss_test = mseloss()
    optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-7)
    lr_schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20, 30, 40, 50, 60,70, 80, 90], gamma=0.5)

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
        trainloss = 0
        for index, (x_train, y_train) in enumerate(tqdm(train_dl)):

            x_train = x_train.to(device)
            y_train = y_train.to(device)
            out = model(x_train)
            loss = criterion(out, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            trainloss += loss.item()

        train_loss.append(loss.item())
        loss_mean = trainloss / len(train_dl)
        print('Train Epoch: {}\t Loss: {:.6f}'.format(epoch, loss_mean))

        model.eval()
        with torch.no_grad():
            X_val = X_val.to(device)
            Y_val = Y_val.to(device)
            pred = model(X_val)
            vloss = loss_test(pred, Y_val).item()

        valid_loss.append(vloss)  
        print('Val set: Loss: {:.4f}'.format(vloss))
        print('learning rate:',optimizer.state_dict()['param_groups'][0]['lr'])
        print('\n')

        lr_schedule.step()

        checkpoint = {
            'net': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'lr_schedule': lr_schedule.state_dict()
        }
        
     
        if epoch%10==9:
            # if not os.path.isdir(cfg.filename):
            #     os.mkdir(cfg.filename)
            # torch.save(checkpoint, cfg.filename+'/ckpt_best_%s.pth' % (str(epoch)))
            SNR = [0, 5, 10, 15, 20, 25, 30]
            for i in range(len(SNR)):
            
                X_t = X_test[5000*i:5000*(i+1),:,:].to(device)
                Y_t = Y_test[5000*i:5000*(i+1),:,:].to(device)

                model.eval()
                with torch.no_grad():
                    tpred  = model(X_t)
                    tloss = loss_test(tpred, Y_t).item()
                
                print('SNR: {} MSE: {:.4f}'.format(SNR[i], tloss))
    np.save('ComplexLight_train_loss.npy',train_loss) 
    np.save('ComplexLight_valid_loss.npy',valid_loss)
