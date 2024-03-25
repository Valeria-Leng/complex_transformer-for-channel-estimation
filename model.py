import torch
from torch import nn
from complexPyTorch.complexLayers import ComplexConv2d, ComplexLinear
from complexPyTorch.complexLayers import ComplexDropout, NaiveComplexBatchNorm2d
from complexPyTorch.complexLayers import ComplexBatchNorm1d
from complexPyTorch.complexFunctions import complex_relu
from torch.nn.functional import softmax, relu, sigmoid
from torchsummary import summary

class ComplexConv1d(nn.Module):

    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding = 0,
                 dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv_r = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self,x):
        input_r = x.real
        input_i = x.imag
        out_r = self.conv_r(input_r)-self.conv_i(input_i)
        out_i = self.conv_r(input_i)+self.conv_i(input_r)

        return out_r + 1j * out_i
    
class Complex_transformer(nn.Module):
    def __init__(self, input_dim =72, embed_dim = 216, out_dim=1008, fc_dim=72, num_heads=2, device='cuda'):
        super().__init__()
        self.dlconv = ComplexConv1d(in_channels=1, out_channels=2,kernel_size=3, padding=1)
        # self.relu = complex_relu()
        self.encoder = Transformer_Encoder(input_dim =input_dim, embed_dim = embed_dim , num_heads=num_heads, device=device)
        self.decoder = Transformer_Decoder(input_dim =input_dim,  out_dim=out_dim, device=device)
        # self.outconv = ComplexConv1d(in_channels=2, out_channels=1, kernel_size=3, padding=1)

    def forward(self, input):
        #input: B, 1, 72, 
        x0 = self.dlconv(input) # B, 2, 72 -> B, 2, 72
        x0 = complex_relu(x0)
        encoder1 = self.encoder(x0) #B, 2, 72
        out = self.decoder(encoder1) #B, 2, 1008 #B, (2-->1), 1008

        return out

class Transformer_Encoder(nn.Module):
    def __init__(self, input_dim =72, embed_dim = 216, num_heads=4, device='cuda'):
        super(Transformer_Encoder, self).__init__()
        self.attention = attention(input_dim=input_dim, embed_dim=embed_dim, num_heads=num_heads, device=device)
        self.FeedforwardNN = FeeaforwardNN(input_dim=input_dim, fc_dim=input_dim)
        self.layernorms = nn.ModuleList([Complex_LayerNorm(input_dim) for _ in range(2)])

    def forward(self, src):
        x1 = self.attention(src) + src
        x1 = self.layernorms[0](x1)
        x2 = self.FeedforwardNN(x1) + x1
        x2 = self.layernorms[1](x2)
        return x2 #B, C, 72

class Transformer_Decoder(nn.Module):
    def __init__(self,input_dim=72,  out_dim=74*14, device='cuda'):
        super(Transformer_Decoder, self).__init__()
        self.conv1 = ComplexConv1d(in_channels=2, out_channels=1, kernel_size=3, padding=1)
        self.conv2 = ComplexConv1d(in_channels=1, out_channels=2, kernel_size=3, padding=1)
        self.conv3 = ComplexConv1d(in_channels=2, out_channels=2, kernel_size=3, padding=1)
        self.conv4 = ComplexConv1d(in_channels=2, out_channels=1, kernel_size=3, padding=1)
        self.layernorm = Complex_LayerNorm(input_dim)
        self.fc = ComplexLinear(input_dim, out_dim)
    def forward(self, src):
        #B, C, 72
        x1 = self.conv1(src) #B, C, 72
        x2 = self.conv2(x1) #B, C, 72
        x3 = complex_relu(x2)
        x4 = self.conv3(x3) + x1
        y = self.layernorm(x4)
        y = self.conv4(self.fc(y)).squeeze() #B, C, 1008
        return y #B, C, 1008

class attention(nn.Module):
    def __init__(self, input_dim =72, embed_dim=216, num_heads = 2, device='cuda') -> None:
        super().__init__()
        self.fc1 = ComplexLinear(input_dim, embed_dim, bias=False)
        self.proj_Q = ComplexLinear(embed_dim, input_dim, bias=False)
        self.proj_K = ComplexLinear(embed_dim, input_dim, bias=False)
        self.proj_V = ComplexLinear(embed_dim, input_dim, bias=False)
        self.out_proj = ComplexLinear(input_dim, input_dim, bias=False)
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.val_dim = input_dim // num_heads
        assert self.val_dim * self.num_heads == self.input_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.val_dim ** -0.5 #4
    def forward(self, src):
        B, C, f = src.shape #B, 2, 72
        src = self.fc1(src) #B, 2, 216
        ###########################################
        #multi_head_attention
        #split Q,K,V
        # Q = self.proj_Q(src).view(B, C, self.num_heads, -1).contiguous().permute(2, 0, 3, 1)  #2, B, 36, C
        # K = self.proj_K(src).view(B, C, self.num_heads, -1).contiguous().permute(2, 0, 3, 1)  #2, B, 36, C
        # V = self.proj_V(src).view(B, C, self.num_heads, -1).contiguous().permute(2, 0, 3, 1)  #2, B, 36, C
        Q = src[:,:,0:self.input_dim].view(B, C, self.num_heads, -1).contiguous().permute(2, 0, 3, 1)
        K = src[:,:,self.input_dim: self.input_dim*2].view(B, C, self.num_heads, -1).contiguous().permute(2, 0, 3, 1)
        V = src[:,:,self.input_dim*2: self.input_dim*3].view(B, C, self.num_heads, -1).contiguous().permute(2, 0, 3, 1)
        #softmax
        attn_weights = self.scaling * torch.matmul(Q, torch.conj_physical(K).transpose(-1, -2)) #2, B, 36, 36
        attn_weights = self.softmax_real(attn_weights) #2, B, 36, 36 dim=-1
        attn = torch.matmul(attn_weights, V) ##2, B, 36, C 
        heads = attn.permute(1, 3, 0, 2).contiguous().view(B, -1, self.num_heads*self.val_dim) #B, C, 72
        heads_all = self.out_proj(heads) #B, C, 72
        return heads_all #B, C, 72
    
    def softmax_real(self, input, attn_mask=None):
        # if real:
        real = torch.real(input)
        # else:
        # real = 10*torch.cos(input.angle())
        if attn_mask is not None:
            real += attn_mask.unsqueeze(0).real.to(self.device)
        # abso[abso == float('inf')] = -abso[abso == float('inf')]
        return softmax(real, dim=-1).type(torch.complex64) ############
class Complex_LayerNorm(nn.Module):

    def __init__(self, embed_dim=None, eps=1e-05, elementwise_affine=True, device='cuda'):
        super().__init__()
        assert not(elementwise_affine and embed_dim is None), 'Give dimensions of learnable parameters or disable them'
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.embed_dim = embed_dim
            self.register_parameter(name='weights', param=torch.nn.Parameter(torch.empty([2, 2], dtype=torch.complex64)))
            self.register_parameter(name='bias', param=torch.nn.Parameter(torch.zeros(embed_dim, dtype=torch.complex64)))
            self.weights = torch.nn.Parameter(torch.eye(2))
            self.weights = torch.nn.Parameter((torch.Tensor([1, 1, 0]).repeat([embed_dim, 1])).unsqueeze(-1))
            self.bias = torch.nn.Parameter(torch.zeros([1, 1, embed_dim], dtype=torch.complex64))
        self.eps = eps

    def forward(self, input):

        ev = torch.unsqueeze(torch.mean(input, dim=-1), dim=-1)
        var_real = torch.unsqueeze(torch.unsqueeze(torch.var(input.real, dim=-1), dim=-1), dim=-1)
        var_imag = torch.unsqueeze(torch.unsqueeze(torch.var(input.imag, dim=-1), dim=-1), dim=-1)

        input = input - ev
        cov = torch.unsqueeze(torch.unsqueeze(torch.mean(input.real * input.imag, dim=-1), dim=-1), dim=-1)
        cov_m_0 = torch.cat((var_real, cov), dim=-1)
        cov_m_1 = torch.cat((cov, var_imag), dim=-1)
        cov_m = torch.unsqueeze(torch.cat((cov_m_0, cov_m_1), dim=-2), dim=-3)
        in_concat = torch.unsqueeze(torch.cat((torch.unsqueeze(input.real, dim=-1), torch.unsqueeze(input.imag, dim=-1)), dim=-1), dim=-1)

        cov_sqr = self.sqrt_2x2(cov_m).cuda()

        # out = self.inv_2x2(cov_sqr).matmul(in_concat)  # [..., 0]
        if self.elementwise_affine:
            real_var_weight = (self.weights[:, 0, :] ** 2).unsqueeze(-1).unsqueeze(0)
            imag_var_weight = (self.weights[:, 1, :] ** 2).unsqueeze(-1).unsqueeze(0)
            cov_weight = (torch.sigmoid(self.weights[:, 2, :].unsqueeze(-1).unsqueeze(0)) - 0.5) * 2 * torch.sqrt(real_var_weight * imag_var_weight)
            weights_mult = torch.cat([torch.cat([real_var_weight, cov_weight], dim=-1), torch.cat([cov_weight, imag_var_weight], dim=-1)], dim=-2).unsqueeze(0).cuda()
            mult_mat = self.sqrt_2x2(weights_mult).matmul(self.inv_2x2(cov_sqr))
            out = mult_mat.matmul(in_concat)  # makes new cov_m = self.weights
        else:
            out = self.inv_2x2(cov_sqr).matmul(in_concat)  # [..., 0]
        out = out[..., 0, 0] + 1j * out[..., 1, 0]  # torch.complex(out[..., 0], out[..., 1]) not used because of memory requirements
        if self.elementwise_affine:
            return out + self.bias.cuda()
        return out

    def inv_2x2(self, input):
        a = torch.unsqueeze(torch.unsqueeze(input[..., 0, 0], dim=-1), dim=-1)
        b = torch.unsqueeze(torch.unsqueeze(input[..., 0, 1], dim=-1), dim=-1)
        c = torch.unsqueeze(torch.unsqueeze(input[..., 1, 0], dim=-1), dim=-1)
        d = torch.unsqueeze(torch.unsqueeze(input[..., 1, 1], dim=-1), dim=-1)
        divisor = a * d - b * c
        mat_1 = torch.cat((d, -b), dim=-2)
        mat_2 = torch.cat((-c, a), dim=-2)
        mat = torch.cat((mat_1, mat_2), dim=-1)
        return mat / divisor

    def sqrt_2x2(self, input):
        a = torch.unsqueeze(torch.unsqueeze(input[..., 0, 0], dim=-1), dim=-1)
        b = torch.unsqueeze(torch.unsqueeze(input[..., 0, 1], dim=-1), dim=-1)
        c = torch.unsqueeze(torch.unsqueeze(input[..., 1, 0], dim=-1), dim=-1)
        d = torch.unsqueeze(torch.unsqueeze(input[..., 1, 1], dim=-1), dim=-1)

        s = torch.sqrt(a * d - b * c)  # sqrt(det)
        t = torch.sqrt(a + d + 2 * s)  # sqrt(trace + 2 * sqrt(det))
        # maybe use 1/t * (M + sI) later, see Wikipedia

        return torch.cat((torch.cat((a + s, b), dim=-2), torch.cat((c, d + s), dim=-2)), dim=-1) / t
class FeeaforwardNN(nn.Module):
    def __init__(self,  input_dim=72,fc_dim=72):
        super().__init__()
        self.fc1 = ComplexLinear(input_dim, fc_dim)
        self.fc2 = ComplexLinear(fc_dim, input_dim) 
    def forward(self, src):
        #####################
        # fc1 --> gelu --> fc1
        src = self.fc1(src) #128, 4, 256
        src = complex_relu(src)
        # src = self.dropout(src)
        src = self.fc2(src) #128, 4, 64
        return src
        #####################    

if __name__=='__main__':
    model = Complex_transformer().cuda()
    summary(model, input_size=(1, 72), dtypes=torch.complex64)

     #input = 256, 1, 1, 72
    #output: 256, 1008
    x = torch.tensor(torch.rand((256, 1, 72), dtype=torch.complex64)).cuda() 
    # print(10*torch.tanh(A))
    out = model(x)
    print(out.shape, out)