import torch
from torch import nn
from torch.nn import functional as F
from enum import Enum
import numpy as np

from ortho_utils import torch_expm


class DeformatorType(Enum):
    FC = 1
    LINEAR = 2
    ID = 3
    ORTHO = 4
    PROJECTIVE = 5
    RANDOM = 6
    WSPACE = 7


class LatentDeformator(nn.Module):
    def __init__(self, shift_dim, num_l, layer=False, all_layer=False, input_dim=None, out_dim=None, inner_dim=1024,
                 type=DeformatorType.FC, random_init=False, bias=True, w_var = None):
        super(LatentDeformator, self).__init__()
        self.type = type
        self.shift_dim = shift_dim
        self.input_dim = input_dim if input_dim is not None else np.product(shift_dim)
        self.out_dim = out_dim if out_dim is not None else np.product(shift_dim)
        self.layer = layer
        self.num_l = num_l
        self.w_var = w_var
        self.all_layer = all_layer

        if self.type == DeformatorType.FC:
            self.fc1 = nn.Linear(self.input_dim, inner_dim)
            self.bn1 = nn.BatchNorm1d(inner_dim)
            self.act1 = nn.ELU()

            self.fc2 = nn.Linear(inner_dim, inner_dim)
            self.bn2 = nn.BatchNorm1d(inner_dim)
            self.act2 = nn.ELU()

            self.fc3 = nn.Linear(inner_dim, inner_dim)
            self.bn3 = nn.BatchNorm1d(inner_dim)
            self.act3 = nn.ELU()

            self.fc4 = nn.Linear(inner_dim, self.out_dim)

        elif self.type in [DeformatorType.LINEAR, DeformatorType.PROJECTIVE]:
            self.linear = nn.Linear(self.input_dim, self.out_dim, bias=bias)
            self.linear.weight.data = torch.zeros_like(self.linear.weight.data)

            min_dim = int(min(self.input_dim, self.out_dim))
            self.linear.weight.data[:min_dim, :min_dim] = torch.eye(min_dim)
            if random_init:
                self.linear.weight.data = 0.1 * torch.randn_like(self.linear.weight.data)

        elif self.type == DeformatorType.ORTHO:
            print("ORTHO")
            assert self.input_dim == self.out_dim, 'In/out dims must be equal for ortho'
            self.log_mat_half = nn.Parameter((1.0 if random_init else 0.001) * torch.randn(
                [self.input_dim, self.input_dim], device='cuda'), True)

        elif self.type == DeformatorType.WSPACE:
            print("WSPACE")
            self.w_linear = nn.Linear(self.input_dim, self.out_dim * self.num_l, bias=bias)
            self.w_linear.weight.data = torch.zeros_like(self.w_linear.weight.data)

            min_dim = int(min(self.input_dim, self.out_dim * self.num_l))
            self.w_linear.weight.data[:min_dim, :min_dim] = torch.eye(min_dim)

        elif self.type == DeformatorType.RANDOM:
            self.linear = torch.empty([self.out_dim, self.input_dim])
            nn.init.orthogonal_(self.linear)

    def forward(self, input):
        if self.type == DeformatorType.ID:
            return input      
        input = input.view([-1, self.input_dim])
        if self.type == DeformatorType.FC:
            x1 = self.fc1(input)
            x = self.act1(self.bn1(x1))

            x2 = self.fc2(x)
            x = self.act2(self.bn2(x2 + x1))

            x3 = self.fc3(x)
            x = self.act3(self.bn3(x3 + x2 + x1))

            out = self.fc4(x) + input
        elif self.type == DeformatorType.LINEAR:
            out  = self.linear(input)

                
        elif self.type == DeformatorType.PROJECTIVE:
            input_norm = torch.norm(input, dim=1, keepdim=True)
            out = self.linear(input)
            #ut = (input_norm / torch.norm(out, dim=1, keepdim=True)) * out
            if self.w_var is not None:
                out = (1 / torch.norm(out, dim=1, keepdim=True)) * out
            else:
                out = (input_norm / torch.norm(out, dim=1, keepdim=True)) * out
        elif self.type == DeformatorType.ORTHO:
            mat = torch_expm((self.log_mat_half - self.log_mat_half.transpose(0, 1)).unsqueeze(0))
            out = F.linear(input, mat)
        elif self.type == DeformatorType.RANDOM:
            self.linear = self.linear.to(input.device)
            out = F.linear(input, self.linear)

        elif self.type == DeformatorType.WSPACE:
            # which = torch.nonzero(input)[:, 1]
            # if which.size() == torch.Size([0]):
            #     return torch.zeros([len(input), self.num_l, self.out_dim], device=out.device)
            #input_norm = torch.norm(input, dim=1, keepdim=True)
            out = self.w_linear(input)
            # out = ((self.input_norm) / torch.norm(out, dim=1, keepdim=True)) * out
            out = out.reshape(len(input), self.num_l, self.out_dim)

        flat_shift_dim = np.product(self.shift_dim)


        # if self.w_var is not None:
        #     magnitude = input_norm
        #     print(self.w_var.shape)
        #     for l in range(len(input)):
        #         variation = out_1[l, :] @ self.w_var @ out_1[l, :].T
        #         sd = torch.sqrt(variation)
        #         out_1[l, :] = out_1[l, :] * magnitude[l, :] * sd
        #     out = out_1
            # av_sd = 0.5555555555
            # comp = sd / av_sd
            #print(sd, comp)
            
        
        if self.all_layer:
            #print('all_layer')
            which = torch.nonzero(input)[:, 1]
            if which.size() == torch.Size([0]):
                return torch.zeros([len(input), self.num_l, self.out_dim], device=out.device)

            allout = torch.zeros([len(input), self.num_l, self.out_dim], device=out.device)
            #out = out.unsqueeze(1).repeat(1, self.num_l, 1)
            for l in range(len(input)):
                c_dir = which[l]
                categ = torch.floor(c_dir / (self.input_dim / (self.num_l / 2))) * 2
                #print(c_dir, self.input_dim, self.num_l, categ)
                # print(out[0, :].shape)
                # print(categ)
                # print(allout[0, 0, :].shape)
                allout[l, categ.to(torch.int32), :] = out[l, :]
                allout[l, categ.to(torch.int32)+1, :] = out[l, :]
            out = allout

        if len(out.shape) > 2:
            if out.shape[-1] < flat_shift_dim:
                #print('correction')
                padding = torch.zeros([out.shape[0], self.num_l, flat_shift_dim - out.shape[-1]], device=out.device)
                out = torch.cat([out, padding], dim=2)
            elif out.shape[-1] > flat_shift_dim:
                #print('correction')
                out = out[:, :, :flat_shift_dim]
        else:
            if out.shape[-1] < flat_shift_dim:
                #print('correction')
                padding = torch.zeros([out.shape[0], flat_shift_dim - out.shape[-1]], device=out.device)
                out = torch.cat([out, padding], dim=1)
            elif out.shape[-1] > flat_shift_dim:
                #print('correction')
                out = out[:, :flat_shift_dim]

        if self.layer:
            #torch.autograd.set_detect_anomaly(True)
            #print('multi_layer')
            which = torch.nonzero(input)[:, 1]
            if which.size() == torch.Size([0]):
                return torch.zeros([len(input), self.num_l, 512], device=out.device)
            
            out = out.unsqueeze(1).repeat(1, self.num_l, 1)
            for l in range(len(input)):
                
                if self.w_var is not None:
                    #print('wvar')
                    variation = out[l, 0, :].cpu() @ self.w_var.cpu() @ out[l, 0, :].T.cpu()
                    sd = torch.sqrt(variation)
                    out[l, :, :] = out[l, :, :] * input_norm[l, :] * sd

                if which[l] < self.input_dim / 3:
                    out[l, 4:, :] *= 0
                elif which[l] < (self.input_dim / 3) * 2:
                    out[l, :4, :] *= 0
                    out[l, 8:, :] *= 0
                else:
                    out[l, :8, :] *= 0


        #handle spatial shifts
        try:
            out = out.view([-1] + self.shift_dim)
        except Exception:
            pass
        #print(out.shape)
        return out


def normal_projection_stat(x):
    x = x.view([x.shape[0], -1])
    direction = torch.randn(x.shape[1], requires_grad=False, device=x.device)
    direction = direction / torch.norm(direction)
    projection = torch.matmul(x, direction)

    std, mean = torch.std_mean(projection)
    return std, mean
