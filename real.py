import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
from torch.nn.functional import softmax, relu, sigmoid, gelu


class ChannelFromer(nn.Module):
    def __init__(self, input_dim =72, embed_dim = 216, out_dim=1008, num_heads=2, device='cuda'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(1, 3), padding=(0, 1))
        self.encoder = Encoder(input_dim, embed_dim, num_heads, device='cuda')
        self.decoder = decoder(input_dim, out_dim, device='cuda')



    def forward(self, input):
        x = self.conv(input)
        x = relu(x)
        x = self.encoder(x)
        out = self.decoder(x)

        return out


class Encoder(nn.Module):
    def __init__(self, input_dim =72, embed_dim = 216, num_heads=2, device='cuda'):
        super().__init__()
        # self.proj = nn.Linear(input_dim, embed_dim)
        self.attention = attention(input_dim, embed_dim, num_heads=2, device='cuda')
        self.ffn = ffn(input_dim)
        self.layernorm = nn.LayerNorm(input_dim)

    def forward(self, x):
        x = self.attention(x) + x
        x = self.layernorm(x)
        x = x + self.ffn(x)
        x = self.layernorm(x)

        return x

class attention(nn.Module):
    def __init__(self, input_dim=72, embed_dim=216, num_heads=2, device='cuda') -> None:
        super().__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.proj = nn.Linear(input_dim, embed_dim)
        self.val_dim = input_dim // num_heads
        self.scaling =  self.val_dim ** -0.5 #6
        self.out_proj = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        #B, C, 2, 72
        B, C, RI, dim = x.shape
        src = self.proj(x)
        Q = src[:,:,:,0:self.input_dim].view(B*C, RI, self.num_heads, -1).contiguous().permute(2, 0, 3, 1) #H, B*C, 36, 2
        K = src[:,:,:,self.input_dim: self.input_dim*2].view(B*C, RI, self.num_heads, -1).contiguous().permute(2, 0, 3, 1)
        V = src[:,:,:,self.input_dim*2: self.input_dim*3].view(B*C, RI, self.num_heads, -1).contiguous().permute(2, 0, 3, 1)
        #softmax
        attn_weights = self.scaling * torch.matmul(Q, K.transpose(-1, -2)) #2, B, 36, 36
        attn_weights = softmax(attn_weights, dim=-1) #2, B, 36, 36 dim=-1
        attn = torch.matmul(attn_weights, V) ##2, B, 36, C 
        heads = attn.permute(1, 3, 0, 2).contiguous().view(B, C, RI, self.num_heads*self.val_dim) #B, C, 72
        heads_all = self.out_proj(heads) #B, C, RI, 72
        return heads_all #B, C, RI, 72
    
class ffn(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, input_dim)
    def forward(self, x):
        out = self.fc(gelu(self.fc(x)))
        return out

class decoder(nn.Module):
    def __init__(self, input_dim = 72, out_dim=1008, device='cuda'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1, 3), padding=(0, 1))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(1, 3), padding=(0, 1))
        self.conv3 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(1, 3), padding=(0, 1))
        self.layernorm = nn.LayerNorm(input_dim)
        self.outproj = nn.Linear(input_dim, out_dim)
        self.outconv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1, 3), padding=(0, 1))

    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.conv3(relu(self.conv2(x0))) + x0
        x1 = self.layernorm(x1)
        x2 = self.outproj(x1)
        out = self.outconv(x2).squeeze()
        return out

if __name__=='__main__':
    model = ChannelFromer().cuda()
    summary(model, input_size=(1, 2, 72), dtypes=torch.float)
    #input = 256,  2, 72
    #output: 256, 2, 1008
    x = torch.tensor(torch.rand((256, 1, 2, 72), dtype=torch.float32)).cuda() 
    # print(10*torch.tanh(A))
    out = model(x)
    print(out.shape, out) #torch.Size([256, 2, 1008])
