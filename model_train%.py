import torch
import torch.nn as nn
import torch.optim as optim
from scipy import io
from config import get_cfg
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
# from model import Complex_transformer
from complex_light import Complex_transformer
from real import ChannelFromer
import os
from tqdm import tqdm
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
        X_test_mat = io.loadmat('data/doppler/test_data.mat')
        X_test = torch.tensor(X_test_mat['test_data'], dtype=torch.complex64).transpose(-1,-2) #(11, 5000, 36, 2)
        Y_test_mat = io.loadmat('data/doppler/test_y.mat')
        Y_test = torch.tensor(Y_test_mat['test_y'], dtype=torch.complex64).transpose(-1,-2) 
    else:
        #dataloader
        X_test_mat = io.loadmat('data/new/new_Test_X.mat')
        # print(X_train_mat['Training_X'].shape)
        X_test = torch.tensor(X_test_mat['Training_X'], dtype=torch.complex64).permute(2, 1, 0).contiguous()
        Y_test_mat = io.loadmat('data/new/new_Test_Y.mat')
        Y_test = torch.tensor(Y_test_mat['Training_Y'], dtype=torch.complex64).permute(2, 1, 0).contiguous()

    #model 
    model = Complex_transformer(input_dim =X_test.shape[-1], out_dim=Y_test.shape[-2], fc_dim=Y_test.shape[-1], num_heads=2).cuda()
    #input = 256, 2, 36
    #output: 256, 14, 72
    criterion = huberloss()

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
            
            print('Doppler: {} Loss: {:.4f}'.format(i, 2*tloss))

    else:
        SNR = [0, 5, 10, 15, 20, 25, 30]
        for i in range(len(SNR)):
        
            X_t = X_test[5000*i:5000*(i+1),:,:].to(device)
            Y_t = Y_test[5000*i:5000*(i+1),:,:].to(device)

            model.eval()
            with torch.no_grad():
                tpred  = model(X_t)
                tloss = criterion(tpred, Y_t).item()
            
            print('SNR: {} MSE: {:.4f}'.format(SNR[i], 2*tloss))

def real_train_percentage():
     #dataloader
    X_train_mat = io.loadmat('data/Training_X.mat')
    # print(X_train_mat['Training_X'].shape)
    X_train = torch.tensor(X_train_mat['Training_X'], dtype=torch.float32).permute(3, 2, 0, 1).contiguous() #torch.Size([118750, 1, 72, 2])
    
    Y_train_mat = io.loadmat('data/Training_Y.mat')
    Y_train = torch.tensor(Y_train_mat['Training_Y'], dtype=torch.float32).permute(3, 2, 0, 1).contiguous()
    # Y_train = torch.view_as_complex(Y_train).view(Y_train.shape[0], cfg.Nf, cfg.Ns) #118750, 72, 14

    X_val_mat = io.loadmat('data/Validation_X.mat')
    # print(X_train_mat['Validation_X'].shape)
    X_val = torch.tensor(X_val_mat['Validation_X'], dtype=torch.float32).permute(3, 2, 0, 1).contiguous()
    
    Y_val_mat = io.loadmat('data/Validation_Y.mat')
    Y_val = torch.tensor(Y_val_mat['Validation_Y'], dtype=torch.float32).permute(3, 2, 0, 1).contiguous()
    # Y_val = torch.view_as_complex(Y_val).view(Y_val.shape[0], cfg.Nf, cfg.Ns) #N, 72, 14
    
    X_train = X_train.transpose(-1, -2) #torch.Size([118750, 1, 2, 72])
    X_val = X_val.transpose(-1, -2) 
    Y_train = Y_train.transpose(-1, -2).squeeze()  #torch.Size([118750, 2, 1008])
    Y_val = Y_val.transpose(-1, -2).squeeze()
    model = ChannelFromer(input_dim=cfg.Nf, embed_dim=3*cfg.Nf, out_dim=cfg.Nf*cfg.Ns, num_heads=2).cuda() 
    #input = 256, 1, 2, 72
    #output: 256, 2, 1008

    SNR = [5, 10, 15, 20, 25]
    percentage = 1
    X_train_all_SNR = []
    Y_train_all_SNR = []
    
    if percentage < 1:
        for i in range(len(SNR)):

            X_SNR = X_train[23750*i:23750*(i+1),:,:,:].to(device) #torch.Size([23750, 2, 36])
            Y_SNR = Y_train[23750*i:23750*(i+1),:,:].to(device)
    
            X_split, _, Y_split, _ = train_test_split(X_SNR, Y_SNR, test_size=1-percentage, random_state=42) #torch.Size([23750*%, 2, 36])
            X_train_all_SNR.append(X_split)
            Y_train_all_SNR.append(Y_split)
        X_train_all_SNR = torch.stack(X_train_all_SNR).view(-1, 1, 2, 72).contiguous()
        Y_train_all_SNR = torch.stack(Y_train_all_SNR).view(-1, 2, 1008).contiguous()
        TrainDataset = TensorDataset(X_train_all_SNR, Y_train_all_SNR)
    else:
        TrainDataset = TensorDataset(X_train, Y_train)

        
    train_dl = DataLoader(TrainDataset, batch_size=cfg.batch_size, shuffle=True)
    criterion = huberloss()
    loss_test = mseloss()
    optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-7)
    # lr_schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 12], gamma=0.1)
    # if cfg.test_flag:
    #     checkpoint = torch.load(cfg.path_checkpoint, map_location=torch.device(device))  # 加载断点
    #     model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
    #     optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
    #     start_epoch = checkpoint['epoch']  # 设置开始的epoch
    #     lr_schedule.load_state_dict(checkpoint['lr_schedule'])
    #     #  lr_schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90,97],gamma=0.1,last_epoch=0)
    #     print('Load epoch {} successfully'.format(start_epoch))

    # else:
    #     start_epoch = -1
    #     print('From epoch 0')
    #     print("------------------------------Ready for training!-------------------------")
    #     # print('L:{}\t M:{}\t N:{}\t K:{}\t Complex?:{}\t'.format(cfg.L, cfg.M, cfg.N, cfg.K, cfg.complex))
    for epoch in range(cfg.epoch):
        model.train()
        train_loss = 0
        for index, (x_train, y_train) in enumerate(tqdm(train_dl)):

            x_train = x_train.to(device)
            y_train = y_train.to(device)
            out = model(x_train)
            loss = 2*criterion(out, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        loss_mean = train_loss / len(train_dl)
        print('Train Epoch: {}\t Loss: {:.6f}'.format(epoch, 2*loss_mean))

        model.eval()
        with torch.no_grad():
            X_val = X_val.to(device)
            Y_val = Y_val.to(device)
            pred = model(X_val)
            vloss = loss_test(pred, Y_val).item()
            
        print('Val set: Loss: {:.4f}'.format(vloss))
        print('learning rate:',optimizer.state_dict()['param_groups'][0]['lr'])
        print('\n')

        # lr_schedule.step()

        # checkpoint = {
        #     'net': model.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        #     'epoch': epoch,
        #     'lr_schedule': lr_schedule.state_dict()
        # }
    X_test_mat = io.loadmat('data/Test_X.mat')
    # print(X_train_mat['Training_X'].shape)
    X_test = torch.tensor(X_test_mat['Training_X'], dtype=torch.float32).permute(3, 2, 0, 1).contiguous()
    # X_test = torch.view_as_complex(X_test).unsqueeze(dim=1) #N, 1, 1, 72
    Y_test_mat = io.loadmat('data/Test_Y.mat')
    Y_test = torch.tensor(Y_test_mat['Training_Y'], dtype=torch.float32).permute(3, 2, 0, 1).contiguous()
    # Y_train = torch.view_as_complex(Y_train).view(Y_train.shape[0], cfg.Nf, cfg.Ns) #118750, 72, 14
    # Y_test = torch.view_as_complex(Y_test).squeeze() #118750, 1008
    
    # print('real!\n')
    X_test = X_test.transpose(-1, -2) #torch.Size([35000, 1, 2, 72])
    Y_test = Y_test.transpose(-1, -2).squeeze()
        

    SNR_test = [0, 5, 10, 15, 20, 25, 30]
    tloss = 0
    for i in range(len(SNR_test)):
    
        X_t = X_test[5000*i:5000*(i+1),:,:].to(device)
        Y_t = Y_test[5000*i:5000*(i+1),:].to(device)


        model.eval()
        with torch.no_grad():
            tpred  = model(X_t) 
            tloss += 2*loss_test(tpred, Y_t).item()
        print('SNR: {} MSE: {:.4f}'.format(SNR_test[i], 2*loss_test(tpred, Y_t)))
    print('percentage: {}, Average_SNR_Loss: {:.4f}'.format(percentage, tloss/len(SNR_test)))
if __name__ == '__main__':
    # real_train_percentage()
    # test()
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
    SNR = [5, 10, 15, 20, 25]
    percentage = 1
    X_train_all_SNR = []
    Y_train_all_SNR = []
    if percentage < 1:
        for i in range(len(SNR)):

            X_SNR = X_train[23750*i:23750*(i+1),:,:].to(device) #torch.Size([23750, 2, 36])
            Y_SNR = Y_train[23750*i:23750*(i+1),:,:].to(device)
    
            X_split, _, Y_split, _ = train_test_split(X_SNR, Y_SNR, test_size=1-percentage, random_state=42) #torch.Size([23750*%, 2, 36])
            X_train_all_SNR.append(X_split)
            Y_train_all_SNR.append(Y_split)
        X_train_all_SNR = torch.stack(X_train_all_SNR).view(-1, 2, 36).contiguous()
        Y_train_all_SNR = torch.stack(Y_train_all_SNR).view(-1, 14, 72).contiguous()
        TrainDataset = TensorDataset(X_train_all_SNR, Y_train_all_SNR)
    else:
        TrainDataset = TensorDataset(X_train, Y_train)
    # ValDataset = TensorDataset(X_val, Y_val)
    # val_dl = DataLoader(ValDataset, batch_size=cfg.batch_size, shuffle=True)
    #model
    model = Complex_transformer(input_dim =X_train.shape[-1], out_dim=Y_train.shape[-2], fc_dim=Y_train.shape[-1], num_heads=2).cuda()    
    #input = 256, 2, 36
    #output: 256, 14, 72
    train_dl = DataLoader(TrainDataset, batch_size=cfg.batch_size, shuffle=True)
    criterion = huberloss()
    loss_test = mseloss()
    optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-7)
    lr_schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 12], gamma=0.1)

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
    for epoch in range(start_epoch+1, cfg.epoch):
        model.train()
        train_loss = 0
        for index, (x_train, y_train) in enumerate(tqdm(train_dl)):

            x_train = x_train.to(device)
            y_train = y_train.to(device)
            out = model(x_train)
            loss = 2*criterion(out, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        loss_mean = train_loss / len(train_dl)
        print('Train Epoch: {}\t Loss: {:.6f}'.format(epoch, loss_mean))

        model.eval()
        with torch.no_grad():
            X_val = X_val.to(device)
            Y_val = Y_val.to(device)
            pred = model(X_val)
            vloss = loss_test(pred, Y_val).item()
            
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
        # if not os.path.isdir(cfg.filename):
        #     os.mkdir(cfg.filename)
        # torch.save(checkpoint, cfg.filename+'/ckpt_best_%s.pth' % (str(epoch)))
    # test()
    X_test_mat = io.loadmat('data/new/new_Test_X.mat')
    # print(X_train_mat['Training_X'].shape)
    X_test = torch.tensor(X_test_mat['Training_X'], dtype=torch.complex64).permute(2, 1, 0).contiguous()
    Y_test_mat = io.loadmat('data/new/new_Test_Y.mat')
    Y_test = torch.tensor(Y_test_mat['Training_Y'], dtype=torch.complex64).permute(2, 1, 0).contiguous()
    SNR = [0, 5, 10, 15, 20, 25, 30]
    # print(X_train_all_SNR.shape[0]/X_train.shape[0])
    tloss = 0.0
    for i in range(len(SNR)):
    
        X_t = X_test[5000*i:5000*(i+1),:,:].to(device)
        Y_t = Y_test[5000*i:5000*(i+1),:,:].to(device)

        model.eval()
        with torch.no_grad():
            tpred  = model(X_t)
            tloss += loss_test(tpred, Y_t).item()

        print('SNR: {} MSE: {:.4f}'.format(SNR[i], loss_test(tpred, Y_t).item()))
    print('percentage: {}, Average_SNR_Loss: {:.4f}'.format(percentage, tloss/len(SNR)))
