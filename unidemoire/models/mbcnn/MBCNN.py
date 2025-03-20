import torch
import torch.nn as nn
import torch.nn.functional as F

from .MBCNN_class import *
import torch.nn.functional as F
import pdb

class MBCNN(nn.Module):
    def __init__(self, nFilters, multi=True):
        super().__init__()
        self.imagesize = 256
        self.sigmoid = nn.Sigmoid()
        self.Space2Depth1 = nn.PixelUnshuffle(2)
        self.Depth2space1 = nn.PixelShuffle(2)

        self.conv_func1 = conv_relu1(12, nFilters * 2, 3, padding=1)
        self.pre_block1 = pre_block((1, 2, 3, 2, 1))
        self.conv_func2 = conv_relu1(128, nFilters * 2, 3, padding=0, stride=2)
        self.pre_block2 = pre_block((1, 2, 3, 2, 1))

        self.conv_func3 = conv_relu1(128, nFilters * 2, 3, padding=0, stride=2)
        self.pre_block3 = pre_block((1, 2, 2, 2, 1))
        self.global_block1 = global_block(self.imagesize // 8)
        self.pos_block1 = pos_block((1, 2, 2, 2, 1))
        self.conv1 = conv1(128, 12, 3,us=[True,False])

        self.conv_func4 = conv_relu1(131, nFilters * 2, 1, padding=0,cat_shape=(3,nFilters*2),set_cat_mul=(False,True))
        self.global_block2 = global_block(self.imagesize // 4)
        self.pre_block4 = pre_block((1, 2, 3, 2, 1))
        self.global_block3 = global_block(self.imagesize // 4)
        self.pos_block2 = pos_block((1, 2, 3, 2, 1))
        self.conv2 = conv1(128, 12, 3,us=[True,False])

        self.conv_func5 = conv_relu1(131, nFilters * 2, 1, padding=0,cat_shape=(3,nFilters*2),set_cat_mul=(False,True))

        self.global_block4 = global_block(self.imagesize // 2)
        self.pre_block5 = pre_block((1, 2, 3, 2, 1))
        self.global_block5 = global_block(self.imagesize // 2)
        self.pos_block3 = pos_block((1, 2, 3, 2, 1))
        self.conv3 = conv1(128, 12, 3,us=[True,False])

    def forward(self, x):
        output_list = []
        shape = list(x.shape)   # [2, 3, 512, 512]
        # batch, channel, height, width = shape
        _x = self.Space2Depth1(x)
        t1 = self.conv_func1(_x)
        t1 = self.pre_block1(t1)
        
        t2 = F.pad(t1, (1, 1, 1, 1))
        t2 = self.conv_func2(t2)
        t2 = self.pre_block2(t2)
        t3 = F.pad(t2, (1, 1, 1, 1))
        t3 = self.conv_func3(t3)
        t3 = self.pre_block3(t3)
        t3 = self.global_block1(t3)
        t3 = self.pos_block1(t3)
        t3_out = self.conv1(t3)
        t3_out = self.Depth2space1(t3_out)
        t3_out = F.sigmoid(t3_out)
        output_list.append(t3_out)

        _t2 = torch.cat([t3_out, t2], dim=-3)
        _t2 = self.conv_func4(_t2)
        _t2 = self.global_block2(_t2)
        _t2 = self.pre_block4(_t2)
        _t2 = self.global_block3(_t2)
        _t2 = self.pos_block2(_t2)
        t2_out = self.conv2(_t2)
        t2_out = self.Depth2space1(t2_out)
        t2_out = F.sigmoid(t2_out)
        output_list.append(t2_out)

        _t1 = torch.cat([t1, t2_out], dim=-3)
        _t1 = self.conv_func5(_t1)
        _t1 = self.global_block4(_t1)
        _t1 = self.pre_block5(_t1)
        _t1 = self.global_block5(_t1)
        _t1 = self.pos_block3(_t1)
        _t1 = self.conv3(_t1)
        y = self.Depth2space1(_t1)

        y = self.sigmoid(y) + torch.Tensor([1e-10]).to(_t1.device)
        output_list.append(y)
        return t3_out,t2_out,y
        #return output_list

# import os
# from torchinfo import summary
# from rich import print
# GPU_ID = 5
# os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % GPU_ID
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# net = MBCNN(64).to(device)
# #print(summary(net, input_size=(1, 3, 512, 512)))
# # print(summary(net))

# # model_stats = summary(
# #     net,
# #     input_size=(1, 3, 512, 512),
# #     verbose=1,
# #     col_names=["kernel_size", "output_size", "num_params"],
# #     row_settings=["var_names"],
# # )
