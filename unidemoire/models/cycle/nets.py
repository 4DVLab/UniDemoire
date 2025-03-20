import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from Models import GM2_UNet5_256, GM2_UNet5_128, GM2_UNet5_64, TMB, Discriminator, L1_ASL

class CycleModel(nn.Module):
    def __init__(self):
        super(CycleModel, self).__init__()

        self.resolution_dict = {
            '256': {'Net_Demoire':'256', 'G_Artifact':'256_2',          'D_moire':'256', 'D_clear':'256'},
            '128': {'Net_Demoire':'128', 'G_Artifact':'128_2',          'D_moire':'128', 'D_clear':'128'},
            '64':  {'Net_Demoire':'64',  'G_Artifact':['64_2', '64_1'], 'D_moire':'64',  'D_clear':'64'},
        }

        self.Net_Demoire = {
            '256': GM2_UNet5_256(6, 3),
            '128': GM2_UNet5_128(6, 3),
            '64': GM2_UNet5_64(3, 3),
            'TMB': TMB(256, 1)
        }
        
        self.G_Artifact = {
            '256_2': GM2_UNet5_256(6, 3),
            '128_2': GM2_UNet5_128(6, 3),
            '64_2': GM2_UNet5_64(3, 3),
            '64_1': TMB(256, 1),
        }

        self.D_moire = {
            '256': Discriminator(6, 256, 256),
            '128': Discriminator(6, 128, 128),
            '64': Discriminator(6, 64, 64),
        }

        self.D_clear = {
            '256': Discriminator(6, 256, 256),
            '128': Discriminator(6, 128, 128),
            '64': Discriminator(6, 64, 64),
        }

        self.downx2 = nn.UpsamplingNearest2d(scale_factor = 0.5)
        self.upx2   = nn.UpsamplingNearest2d(scale_factor = 2)


        # LOSS FUNCTIONS
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_MSE = torch.nn.MSELoss()
        self.criterion_content = L1_ASL()
        self.Loss = L1_ASL()

        # Initialize weights
        for key in self.Net_Demoire.keys():
            self.Net_Demoire[key].apply(self.weights_init)
            
        for key in self.G_Artifact.keys():
            self.G_Artifact[key].apply(self.weights_init)

        for key in self.D_moire.keys():
            self.D_moire[key].apply(self.weights_init)

        for key in self.D_clear.keys():
            self.D_clear[key].apply(self.weights_init)


    # Custom weights initialization called on network
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

    
    def forward(self, MOIRE, CLEAR, historgram, device):

        Tensor = torch.cuda.FloatTensor 

        # load data
        MOIRE_256 = MOIRE
        MOIRE_128 = self.downx2(MOIRE_256)
        MOIRE_64  = self.downx2(MOIRE_128)

        CLEAR_256 = CLEAR
        CLEAR_128 = self.downx2(CLEAR_256)
        CLEAR_64  = self.downx2(CLEAR_128)

        historgram = historgram.float()

        valid_256 = Variable(Tensor(MOIRE_256.size(0), 1, 1, 1).fill_(1.0).to(device), requires_grad=False)
        fake_256  = Variable(Tensor(MOIRE_256.size(0), 1, 1, 1).fill_(0.0).to(device), requires_grad=False)

        valid_128 = Variable(Tensor(MOIRE_128.size(0), 1, 1, 1).fill_(1.0).to(device), requires_grad=False)
        fake_128  = Variable(Tensor(MOIRE_128.size(0), 1, 1, 1).fill_(0.0).to(device), requires_grad=False)

        valid_64 = Variable(Tensor(MOIRE_64.size(0), 1, 1, 1).fill_(1.0).to(device), requires_grad=False)
        fake_64  = Variable(Tensor(MOIRE_64.size(0), 1, 1, 1).fill_(0.0).to(device), requires_grad=False)  

        for resolution in self.resolution_dict.keys():
            
            Net_Demoire  = self.Net_Demoire[self.resolution_dict[resolution]['Net_Demoire']]
            
            if resolution == '64':
                G_Artifact_1 = self.G_Artifact[self.resolution_dict[resolution]['G_Artifact'][0]]
            G_Artifact_2 = self.G_Artifact[self.resolution_dict[resolution]['G_Artifact']] if resolution != '64' else self.G_Artifact[self.resolution_dict[resolution]['G_Artifact'][1]]
            
            D_moire = self.D_moire[self.resolution_dict[resolution]['D_moire']]  
            D_clear = self.D_clear[self.resolution_dict[resolution]['D_clear']]

            