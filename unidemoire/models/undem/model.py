import torch.nn as nn
import torch.nn.functional as F
import torch


def get_pretrain_undem_model(dataset_name="TIP", patch_size=256):
    # Networks
    _, netG_mo_class1, _, _, encoder_mo_class1, _ = define_models()
    _, netG_mo_class2, _, _, encoder_mo_class2, _ = define_models()
    _, netG_mo_class3, _, _, encoder_mo_class3, _ = define_models()
    _, netG_mo_class4, _, _, encoder_mo_class4, _ = define_models()

    if dataset_name == 'UHDM':
        if patch_size == 384:
            netG_mo_class1, encoder_mo_class1 = load_model(netG_mo_class1, encoder_mo_class1,
                ["/inspurfs/group/mayuexin/yangzemin/code/UnDeM/models/fhdmi_fake_384/fhdmi_fake_class1_384/netG_mo.pth",
                 "/inspurfs/group/mayuexin/yangzemin/code/UnDeM/models/fhdmi_fake_384/fhdmi_fake_class1_384/encoder_mo.pth"])
            
            netG_mo_class2, encoder_mo_class2 = load_model(netG_mo_class2, encoder_mo_class2,
                ["/inspurfs/group/mayuexin/yangzemin/code/UnDeM/models/fhdmi_fake_384/fhdmi_fake_class2_384/netG_mo.pth",
                 "/inspurfs/group/mayuexin/yangzemin/code/UnDeM/models/fhdmi_fake_384/fhdmi_fake_class2_384/encoder_mo.pth"])
            
            netG_mo_class3, encoder_mo_class3 = load_model(netG_mo_class3, encoder_mo_class3,
                ["/inspurfs/group/mayuexin/yangzemin/code/UnDeM/models/fhdmi_fake_384/fhdmi_fake_class3_384/netG_mo.pth",
                 "/inspurfs/group/mayuexin/yangzemin/code/UnDeM/models/fhdmi_fake_384/fhdmi_fake_class3_384/encoder_mo.pth"])
            
            netG_mo_class4, encoder_mo_class4 = load_model(netG_mo_class4, encoder_mo_class4,
                ["/inspurfs/group/mayuexin/yangzemin/code/UnDeM/models/fhdmi_fake_384/fhdmi_fake_class4_384/netG_mo.pth",
                 "/inspurfs/group/mayuexin/yangzemin/code/UnDeM/models/fhdmi_fake_384/fhdmi_fake_class4_384/encoder_mo.pth"])
            
        elif patch_size == 192:
            netG_mo_class1, encoder_mo_class1 = load_model(netG_mo_class1, encoder_mo_class1,
                ["/inspurfs/group/mayuexin/yangzemin/code/UnDeM/models/fhdmi_fake_192/fhdmi_fake_class1_192/netG_mo.pth",
                 "/inspurfs/group/mayuexin/yangzemin/code/UnDeM/models/fhdmi_fake_192/fhdmi_fake_class1_192/encoder_mo.pth"])
            
            netG_mo_class2, encoder_mo_class2 = load_model(netG_mo_class2, encoder_mo_class2,
                ["/inspurfs/group/mayuexin/yangzemin/code/UnDeM/models/fhdmi_fake_192/fhdmi_fake_class2_192/netG_mo.pth",
                 "/inspurfs/group/mayuexin/yangzemin/code/UnDeM/models/fhdmi_fake_192/fhdmi_fake_class2_192/encoder_mo.pth"])
            
            netG_mo_class3, encoder_mo_class3 = load_model(netG_mo_class3, encoder_mo_class3,
                ["/inspurfs/group/mayuexin/yangzemin/code/UnDeM/models/fhdmi_fake_192/fhdmi_fake_class3_192/netG_mo.pth",
                 "/inspurfs/group/mayuexin/yangzemin/code/UnDeM/models/fhdmi_fake_192/fhdmi_fake_class3_192/encoder_mo.pth"])
            
            netG_mo_class4, encoder_mo_class4 = load_model(netG_mo_class4, encoder_mo_class4,
                ["/inspurfs/group/mayuexin/yangzemin/code/UnDeM/models/fhdmi_fake_192/fhdmi_fake_class4_192/netG_mo.pth",
                 "/inspurfs/group/mayuexin/yangzemin/code/UnDeM/models/fhdmi_fake_192/fhdmi_fake_class4_192/encoder_mo.pth"])

    elif dataset_name == 'TIP':
        if patch_size == 256:
            netG_mo_class1, encoder_mo_class1 = load_model(netG_mo_class1, encoder_mo_class1,
                ["/inspurfs/group/mayuexin/yangzemin/code/UnDeM/models/tip_fake_256/tip_fake_class1_256/netG_mo.pth",
                 "/inspurfs/group/mayuexin/yangzemin/code/UnDeM/models/tip_fake_256/tip_fake_class1_256/encoder_mo.pth"])
            
            netG_mo_class2, encoder_mo_class2 = load_model(netG_mo_class2, encoder_mo_class2,
                ["/inspurfs/group/mayuexin/yangzemin/code/UnDeM/models/tip_fake_256/tip_fake_class2_256/netG_mo.pth",
                 "/inspurfs/group/mayuexin/yangzemin/code/UnDeM/models/tip_fake_256/tip_fake_class2_256/encoder_mo.pth"])
            
            netG_mo_class3, encoder_mo_class3 = load_model(netG_mo_class3, encoder_mo_class3,
                ["/inspurfs/group/mayuexin/yangzemin/code/UnDeM/models/tip_fake_256/tip_fake_class3_256/netG_mo.pth",
                 "/inspurfs/group/mayuexin/yangzemin/code/UnDeM/models/tip_fake_256/tip_fake_class3_256/encoder_mo.pth"])
            
            netG_mo_class4, encoder_mo_class4 = load_model(netG_mo_class4, encoder_mo_class4,
                ["/inspurfs/group/mayuexin/yangzemin/code/UnDeM/models/tip_fake_256/tip_fake_class4_256/netG_mo.pth",
                 "/inspurfs/group/mayuexin/yangzemin/code/UnDeM/models/tip_fake_256/tip_fake_class4_256/encoder_mo.pth"])


    elif dataset_name == 'FHDMi':
        if patch_size == 192:
            netG_mo_class1, encoder_mo_class1 = load_model(netG_mo_class1, encoder_mo_class1,
                ["/root/UDA_Demoire/UnDeM/models/uhdm_fake_192/uhdm_fake_class1_192/netG_mo.pth",
                 "/root/UDA_Demoire/UnDeM/models/uhdm_fake_192/uhdm_fake_class1_192/encoder_mo.pth"])
            
            netG_mo_class2, encoder_mo_class2 = load_model(netG_mo_class2, encoder_mo_class2,
                ["/root/UDA_Demoire/UnDeM/models/uhdm_fake_192/uhdm_fake_class2_192/netG_mo.pth",
                 "/root/UDA_Demoire/UnDeM/models/uhdm_fake_192/uhdm_fake_class2_192/encoder_mo.pth"])
            
            netG_mo_class3, encoder_mo_class3 = load_model(netG_mo_class3, encoder_mo_class3,
                ["/root/UDA_Demoire/UnDeM/models/uhdm_fake_192/uhdm_fake_class3_192/netG_mo.pth",
                 "/root/UDA_Demoire/UnDeM/models/uhdm_fake_192/uhdm_fake_class3_192/encoder_mo.pth"])
            
            netG_mo_class4, encoder_mo_class4 = load_model(netG_mo_class4, encoder_mo_class4,
                ["/root/UDA_Demoire/UnDeM/models/uhdm_fake_192/uhdm_fake_class4_192/netG_mo.pth",
                 "/root/UDA_Demoire/UnDeM/models/uhdm_fake_192/uhdm_fake_class4_192/encoder_mo.pth"])
        elif patch_size == 384:
            netG_mo_class1, encoder_mo_class1 = load_model(netG_mo_class1, encoder_mo_class1,
                ["/root/UDA_Demoire/UnDeM/models/uhdm_fake_384/uhdm_fake_class1_384/netG_mo.pth",
                 "/root/UDA_Demoire/UnDeM/models/uhdm_fake_384/uhdm_fake_class1_384/encoder_mo.pth"])
            
            netG_mo_class2, encoder_mo_class2 = load_model(netG_mo_class2, encoder_mo_class2,
                ["/root/UDA_Demoire/UnDeM/models/uhdm_fake_384/uhdm_fake_class2_384/netG_mo.pth",
                 "/root/UDA_Demoire/UnDeM/models/uhdm_fake_384/uhdm_fake_class2_384/encoder_mo.pth"])
            
            netG_mo_class3, encoder_mo_class3 = load_model(netG_mo_class3, encoder_mo_class3,
                ["/root/UDA_Demoire/UnDeM/models/uhdm_fake_384/uhdm_fake_class3_384/netG_mo.pth",
                 "/root/UDA_Demoire/UnDeM/models/uhdm_fake_384/uhdm_fake_class3_384/encoder_mo.pth"])
            
            netG_mo_class4, encoder_mo_class4 = load_model(netG_mo_class4, encoder_mo_class4,
                ["/root/UDA_Demoire/UnDeM/models/uhdm_fake_384/uhdm_fake_class4_384/netG_mo.pth",
                 "/root/UDA_Demoire/UnDeM/models/uhdm_fake_384/uhdm_fake_class4_384/encoder_mo.pth"])
        elif patch_size == 768:
            netG_mo_class1, encoder_mo_class1 = load_model(netG_mo_class1, encoder_mo_class1,
                ["/root/UDA_Demoire/UnDeM/models/uhdm_fake_768/uhdm_fake_class1_768/netG_mo.pth",
                 "/root/UDA_Demoire/UnDeM/models/uhdm_fake_768/uhdm_fake_class1_768/encoder_mo.pth"])
            
            netG_mo_class2, encoder_mo_class2 = load_model(netG_mo_class2, encoder_mo_class2,
                ["/root/UDA_Demoire/UnDeM/models/uhdm_fake_768/uhdm_fake_class2_768/netG_mo.pth",
                 "/root/UDA_Demoire/UnDeM/models/uhdm_fake_768/uhdm_fake_class2_768/encoder_mo.pth"])
            
            netG_mo_class3, encoder_mo_class3 = load_model(netG_mo_class3, encoder_mo_class3,
                ["/root/UDA_Demoire/UnDeM/models/uhdm_fake_768/uhdm_fake_class3_768/netG_mo.pth",
                 "/root/UDA_Demoire/UnDeM/models/uhdm_fake_768/uhdm_fake_class3_768/encoder_mo.pth"])
            
            netG_mo_class4, encoder_mo_class4 = load_model(netG_mo_class4, encoder_mo_class4,
                ["/root/UDA_Demoire/UnDeM/models/uhdm_fake_768/uhdm_fake_class4_768/netG_mo.pth",
                 "/root/UDA_Demoire/UnDeM/models/uhdm_fake_768/uhdm_fake_class4_768/encoder_mo.pth"])
            
    return  netG_mo_class1, netG_mo_class2, netG_mo_class3, netG_mo_class4, \
            encoder_mo_class1, encoder_mo_class2, encoder_mo_class3, encoder_mo_class4

def load_model(netG_mo_class, encoder_mo_class, path_list):
    path_1, path_2 = path_list[0], path_list[1]
    #path_1 = './models/fhdmi_fake_192/fhdmi_fake_class1_192/netG_mo.pth' # set your path here
    print('load: ' + path_1)
    netG_mo_class.load_state_dict(torch.load(path_1))

    #path_2 = './models/fhdmi_fake_192/fhdmi_fake_class1_192/encoder_mo.pth' # set your path here
    print('load: ' + path_2)
    encoder_mo_class.load_state_dict(torch.load(path_2))

    return netG_mo_class, encoder_mo_class


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Encoder(nn.Module):
    def __init__(self, init_weights=False):
        super(Encoder, self).__init__()

        # Initial convolution block
        self.conv1_b = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(3, 16, 7),
                                     nn.InstanceNorm2d(16),
                                     nn.ReLU(inplace=True))
        self.conv3_b = nn.Sequential(ResidualBlock(16))
        self.conv4_b = nn.Sequential(ResidualBlock(16))

        if init_weights:
            self.apply(weights_init_normal)

    def forward(self, xin):
        x = self.conv1_b(xin)
        x = self.conv3_b(x)
        x = self.conv4_b(x)
        xout = x
        return xout

class Generator(nn.Module):
    def __init__(self, init_weights=False):
        super(Generator, self).__init__()

        # Initial convolution block
        self.conv1_b = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(6, 128, 7),
                                     nn.InstanceNorm2d(128),
                                     nn.ReLU(inplace=True))
        self.downconv2_b = nn.Sequential(nn.Conv2d(128, 128, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(128),
                                         nn.ReLU(inplace=True))
        self.downconv3_b = nn.Sequential(nn.Conv2d(128, 128, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(128),
                                         nn.ReLU(inplace=True))
        self.conv4_b = nn.Sequential(ResidualBlock(128))
        self.conv5_b = nn.Sequential(ResidualBlock(128))
        self.conv9_b = nn.Sequential(ResidualBlock(128))
        self.conv10_b = nn.Sequential(ResidualBlock(128))
        self.conv11_b = nn.Sequential(ResidualBlock(128))
        self.conv12_b = nn.Sequential(ResidualBlock(128))
        self.upconv13_b = nn.Sequential(nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(128),
                                        nn.ReLU(inplace=True))
        self.upconv14_b = nn.Sequential(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(64),
                                        nn.ReLU(inplace=True))
        self.conv15_b = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, 3, 7))

        if init_weights:
            self.apply(weights_init_normal)

    def forward(self, reference, xin):
        x = torch.cat((reference, xin), 1)
        x = self.conv1_b(x)
        x = self.downconv2_b(x)
        x = self.downconv3_b(x)
        x = self.conv4_b(x)
        x = self.conv5_b(x)
        x = self.conv9_b(x)
        x = self.conv10_b(x)
        x = self.conv11_b(x)
        x = self.conv12_b(x)
        x = self.upconv13_b(x)
        x = self.upconv14_b(x)
        x = self.conv15_b(x)
        xout = x + xin
        return xout.tanh()

class Generator_mo(nn.Module):
    def __init__(self, init_weights=False):
        super(Generator_mo, self).__init__()

        # Initial convolution block
        self.conv1_b = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(19, 128, 7),
                                     nn.InstanceNorm2d(128),
                                     nn.ReLU(inplace=True))
        self.downconv2_b = nn.Sequential(nn.Conv2d(128, 128, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(128),
                                         nn.ReLU(inplace=True))
        self.downconv3_b = nn.Sequential(nn.Conv2d(128, 128, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(128),
                                         nn.ReLU(inplace=True))
        self.conv4_b = nn.Sequential(ResidualBlock(128))
        self.conv5_b = nn.Sequential(ResidualBlock(128))
        self.conv9_b = nn.Sequential(ResidualBlock(128))
        self.conv10_b = nn.Sequential(ResidualBlock(128))
        self.conv11_b = nn.Sequential(ResidualBlock(128))
        self.conv12_b = nn.Sequential(ResidualBlock(128))
        self.upconv13_b = nn.Sequential(nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(128),
                                        nn.ReLU(inplace=True))
        self.upconv14_b = nn.Sequential(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(64),
                                        nn.ReLU(inplace=True))
        self.conv15_b = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, 3, 7))

        if init_weights:
            self.apply(weights_init_normal)

    def forward(self, reference, xin):
        x = torch.cat((reference, xin), 1)
        x = self.conv1_b(x)
        x = self.downconv2_b(x)
        x = self.downconv3_b(x)
        x = self.conv4_b(x)
        x = self.conv5_b(x)
        x = self.conv9_b(x)
        x = self.conv10_b(x)
        x = self.conv11_b(x)
        x = self.conv12_b(x)
        x = self.upconv13_b(x)
        x = self.upconv14_b(x)
        x = self.conv15_b(x)
        xout = x + xin
        return xout.tanh()

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(3, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self,x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1).squeeze() #global avg pool


class Discriminator_two(nn.Module):
    def __init__(self):
        super(Discriminator_two, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(6, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x_1, x_2):
        x = torch.cat((x_1, x_2), 1)
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1).squeeze() #global avg pool


def define_models():
    # Networks
    netG_color = Generator()
    netG_mo = Generator_mo()

    netD_color = Discriminator_two()
    netD_mo = Discriminator()

    encoder_mo = Encoder()
    encoder_content = Encoder()

    netG_color.cuda()
    netG_mo.cuda()
    netD_color.cuda()
    netD_mo.cuda()
    encoder_mo.cuda()
    encoder_content.cuda()

    netG_color.apply(weights_init_normal)
    netG_mo.apply(weights_init_normal)
    netD_color.apply(weights_init_normal)
    netD_mo.apply(weights_init_normal)
    encoder_mo.apply(weights_init_normal)
    encoder_content.apply(weights_init_normal)

    return netG_color, netG_mo, netD_color, netD_mo, encoder_mo, encoder_content


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def define_models_2():
    # Networks
    netG_color = Generator()
    netG_mo = Generator()

    netD_color = Discriminator_two()
    netD_mo = Discriminator()
    netD_Nomo = Discriminator()

    encoder_mo = Encoder()
    encoder_content = Encoder()

    netG_color.cuda()
    netG_mo.cuda()
    netD_color.cuda()
    netD_mo.cuda()
    netD_Nomo.cuda()
    encoder_mo.cuda()
    encoder_content.cuda()

    netG_color.apply(weights_init_normal)
    netG_mo.apply(weights_init_normal)
    netD_color.apply(weights_init_normal)
    netD_mo.apply(weights_init_normal)
    encoder_mo.apply(weights_init_normal)
    encoder_content.apply(weights_init_normal)

    return netG_color, netG_mo, netD_color, netD_mo, netD_Nomo, encoder_mo, encoder_content


class testmodel(nn.Module):
    def __init__(self, netG_mo, encoder_mo):
        super(testmodel, self).__init__()
        self.netG_mo = netG_mo
        self.encoder_mo = encoder_mo
        # self.test_natural = torch.randn(2, 3, 256, 256)

    def forward(self, x, y):
       real_mo_feat = self.encoder_mo(xin=x)
       fake_mo = self.netG_mo(reference=real_mo_feat, xin=y)
       return fake_mo