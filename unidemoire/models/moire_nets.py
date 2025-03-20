import pytorch_lightning as pl
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
from pytorch_lightning import seed_everything
from torch.nn.parameter import Parameter
from torch.optim import lr_scheduler
from functools import partial
import time
import torch.optim as optim

from omegaconf import OmegaConf
import glob

from .shooting.method import Shooting
from .undem.model import get_pretrain_undem_model
from .cycle.networks import CycleModel
from .cycle.Models.Loss_func_demoire import *
from .esdnet.nets import ESDNet
from .mbcnn.MBCNN import MBCNN
from .pmtnet.PMTNet import PMTNet

from unidemoire.util import instantiate_from_config
from .utils.loss_util import *
from .utils.common import *
from .utils.metric import create_metrics

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class Demoireing_Model(pl.LightningModule):
    def __init__(self, model_name, mode, blending_method, blending_model_path, dataset, evaluation_time, evaluation_metric, network_config, loss_config, optimizer_config, save_img, ckpt_path=None, ignore_keys=[]):
        super().__init__()
        self.model_name = model_name
        self.mode = mode
        self.dataset = dataset
        self.evaluation_time = evaluation_time
        self.evaluation_metric = evaluation_metric
        self.network_config = network_config
        self.loss_config = loss_config
        self.optimizer_config = optimizer_config
        self.blending_method = blending_method
        self.model = self.build_up_models()
        self.loss_fn = self.loss_function()
        self.save_img = save_img
        
        if self.mode == 'COMBINE_ONLINE_WITH_CLEAN':
            self.use_additional_clean_image_only = True
            self.mode = 'COMBINE_ONLINE'
        else:
            self.use_additional_clean_image_only = False
        
        if self.mode != 'original':
            if self.blending_method == 'shooting':
                print("----- Shooting Method -----")
                self.moire_blending_model = Shooting
            elif self.blending_method == 'undem_tip' or self.blending_method == 'undem_moire_pattern':
                if self.blending_method == 'undem_tip':
                    self.undem_inference_data = 'tip_real_moire'
                    print("----- UnDeM Method with TIP -----")
                elif self.blending_method == 'undem_moire_pattern':
                    self.undem_inference_data = 'moire_pattern'
                    print("----- UnDeM Method with Moire Pattern -----")
                self.blending_method = 'undem'
                self.moire_blending_model = self.get_undem_model()

            ## TODO: Cycle
            elif self.blending_method == 'cycle_tip' or self.blending_method == 'cycle_moire_pattern':
                if self.blending_method == 'cycle_tip':
                    self.cycle_inference_data = 'tip_real_moire'
                    print("----- Cycle Method with TIP -----")
                elif self.blending_method == 'cycle_moire_pattern':
                    self.cycle_inference_data = 'moire_pattern'
                    print("----- Cycle Method with Moire Pattern -----")
                self.blending_method = 'cycle'
                self.moire_blending_model = None
                assert self.model_name == 'cycle'
                #! Important: This property activates manual optimization.
                self.automatic_optimization = False

            elif self.blending_method == 'unidemoire':
                print("----- UniDemoire -----")
                self.blending_model_path  = blending_model_path
                self.moire_blending_model = self.get_blending_model()
            else:
                print("----- MoireSpace -----")
                self.moire_blending_model = None
        else:
            self.moire_blending_model = None
        
        self.compute_metrics = None
        if self.evaluation_time:
            # metric calculation may have negative impact on inference speed
            self.evaluation_metric = False
        if self.evaluation_metric:
            # load LPIPS metric
            self.compute_metrics = create_metrics(self.dataset, device=self.device)

        if ckpt_path is not None:
            print(f"Loading Checkpoint from {ckpt_path}")
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            
        seed_everything(123)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"Resumed checkpoint contains {len(unexpected)} unexpected keys")

    def val_mode(self, model):
        model = model.eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        return model

    def get_config_from_ckpt_path(self, ckpt_path):
        if os.path.isfile(ckpt_path):              
            # paths = opt.resume.split("/")
            try:
                logdir = '/'.join(ckpt_path.split('/')[:-1])
                # idx = len(paths)-paths[::-1].index("logs")+1
                print(f'Encoder dir is {ckpt_path}')
            except ValueError:
                paths = ckpt_path.split("/")
                idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
                logdir = "/".join(paths[:idx])
            ckpt = ckpt_path
        else:
            assert os.path.isdir(ckpt_path), f"{ckpt_path} is not a directory"
            logdir = ckpt_path.rstrip("/")
            ckpt = os.path.join(logdir, "model.ckpt")

        base_configs = sorted(glob.glob(os.path.join(logdir, "config.yaml")))
        base = base_configs

        configs = [OmegaConf.load(cfg) for cfg in base]
        return configs[0]['model'] 

    def get_undem_model(self):
        #* define undem model & loading checkpoint               
        netG_mo_class1, netG_mo_class2, netG_mo_class3, netG_mo_class4, \
            encoder_mo_class1, encoder_mo_class2, encoder_mo_class3, encoder_mo_class4 = get_pretrain_undem_model(dataset_name='UHDM', patch_size=384)
        return [
            netG_mo_class1.eval(), encoder_mo_class1.eval(),
            netG_mo_class2.eval(), encoder_mo_class2.eval(),
            netG_mo_class3.eval(), encoder_mo_class3.eval(),
            netG_mo_class4.eval(), encoder_mo_class4.eval(),
        ]

    def get_cycle_model(self):
        model = CycleModel(self.device)
        return model

    def get_blending_model(self):
        self.moire_blending_model_config = self.get_config_from_ckpt_path(self.blending_model_path)
        self.moire_blending_model_config['params']['ckpt_path'] = self.blending_model_path

        #* Get blending_model
        moire_blending_model = instantiate_from_config(self.moire_blending_model_config)
        moire_blending_model = self.val_mode(moire_blending_model)
        return moire_blending_model

    def build_up_models(self):
        if self.model_name == "ESDNet":
            model = ESDNet(
                en_feature_num=self.network_config["en_feature_num"],
                en_inter_num=self.network_config["en_inter_num"],
                de_feature_num=self.network_config["de_feature_num"],
                de_inter_num=self.network_config["de_inter_num"],
                sam_number=self.network_config["sam_number"],
            )
            if self.training:
                model._initialize_weights()
        elif self.model_name == "MBCNN":
            model = MBCNN(nFilters=self.network_config["n_filters"])
            
        #! PMTNet (for rebuttal)
        elif self.model_name == "PMTNet":
            model = PMTNet()

        #! CycleModel (for rebuttal)
        elif self.model_name == "cycle":
            model = self.get_cycle_model()
            
        return model

    def loss_function(self):
        if self.model_name == "ESDNet":
            loss_fn = multi_VGGPerceptualLoss(lam=self.loss_config["LAM"], lam_p=self.loss_config["LAM_P"])
        elif self.model_name == "MBCNN":
            loss_fn = MBCNN_Loss()
        elif self.model_name == "PMTNet":
            loss_fn = PMTNet_Loss(device=self.device)
        elif self.model_name == "cycle":
            # LOSS FUNCTIONS
            criterion_GAN = torch.nn.MSELoss()
            criterion_cycle = torch.nn.L1Loss()
            criterion_MSE = torch.nn.MSELoss()
            criterion_content = L1_ASL()
            Loss = L1_ASL()
            loss_fn = [criterion_GAN, criterion_cycle, criterion_MSE, criterion_content, Loss]
        return loss_fn

    def setup_optimizer(self):
        if self.model_name == "ESDNet":
            optimizer = optim.Adam(
                [{'params': self.model.parameters(), 'initial_lr': self.learning_rate}], 
                betas=(self.optimizer_config["beta1"], self.optimizer_config["beta1"]))
        elif self.model_name == "MBCNN":    
            optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate)
        
        elif self.model_name == "PMTNet":    
            optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate)

        elif self.model_name == "cycle":
            # Optimizers
            optimizer_Net_Demoire_256 = torch.optim.AdamW(self.model.Net_Demoire_256.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
            optimizer_Net_Demoire_128 = torch.optim.AdamW(self.model.Net_Demoire_128.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
            optimizer_Net_Demoire_64 = torch.optim.AdamW(self.model.Net_Demoire_64.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
            optimizer_Net_Demoire_TMB = torch.optim.AdamW(self.model.Net_Demoire_TMB.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))

            # 256 size
            optimizer_G_256_2 = torch.optim.AdamW(self.model.G_Artifact_256_2.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
            optimizer_D_Moire_256 = torch.optim.AdamW(self.model.D_Moire_256.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
            optimizer_D_Clear_256 = torch.optim.AdamW(self.model.D_Clear_256.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))

            # 128 size
            optimizer_G_128_2 = torch.optim.AdamW(self.model.G_Artifact_128_2.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
            optimizer_D_Moire_128 = torch.optim.AdamW(self.model.D_Moire_128.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
            optimizer_D_Clear_128 = torch.optim.AdamW(self.model.D_Clear_128.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))

            # 64 size
            optimizer_G_64_1 = torch.optim.AdamW(self.model.G_Artifact_64_1.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
            optimizer_G_64_2 = torch.optim.AdamW(self.model.G_Artifact_64_2.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
            optimizer_D_Moire_64 = torch.optim.AdamW(self.model.D_Moire_64.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
            optimizer_D_Clear_64 = torch.optim.AdamW(self.model.D_Clear_64.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
            
            optimizer = [optimizer_Net_Demoire_256, optimizer_Net_Demoire_128, optimizer_Net_Demoire_64, optimizer_Net_Demoire_TMB,
                         optimizer_G_256_2, optimizer_D_Moire_256, optimizer_D_Clear_256, 
                         optimizer_G_128_2, optimizer_D_Moire_128, optimizer_D_Clear_128,
                         optimizer_G_64_1, optimizer_G_64_2, optimizer_D_Moire_64, optimizer_D_Clear_64]
        
        return optimizer

    def setup_scheduler(self):
        if self.model_name == "ESDNet":
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, 
                T_0=self.optimizer_config["T_0"], 
                T_mult=self.optimizer_config["T_mult"], 
                eta_min=self.optimizer_config["eta_min"],
            )

        elif self.model_name == "MBCNN":
            # Based on MBCNN source code:
            # If the validation loss drops below 0.001 dB for four consecutive Epochs, the learning rate is halved. When the learning rate is below $10^{-6}$, the training process is complete!
            # But the PyTorch replication code for MBCNN omits this part, resulting in other models not using this strategy when using MBCNN for new experiments.
            # Here, to be fair, we can only continue to follow the design of the methods above, i.e., the learning rate is not adjusted when training MBCNNs.
            scheduler = None

        elif self.model_name == "PMTNet":
            scheduler = lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.5)

        elif self.model_name == "cycle":
            # Step LR
            lr_scheduler_Net_256 = optim.lr_scheduler.StepLR(self.optimizer[0], step_size = 100, gamma = 0.1)
            lr_scheduler_Net_128 = optim.lr_scheduler.StepLR(self.optimizer[1], step_size = 100, gamma = 0.1)
            lr_scheduler_Net_64  = optim.lr_scheduler.StepLR(self.optimizer[2], step_size = 100, gamma = 0.1)
            lr_scheduler_Net_Demoire_TMB = optim.lr_scheduler.StepLR(self.optimizer[3], step_size = 100, gamma = 0.1)

            # 256
            lr_scheduler_G_256_2 = optim.lr_scheduler.StepLR(self.optimizer[4], step_size = 100, gamma = 0.1)
            lr_scheduler_D_Moire_256 = optim.lr_scheduler.StepLR(self.optimizer[5], step_size = 100, gamma = 0.1)
            lr_scheduler_D_Clear_256 = optim.lr_scheduler.StepLR(self.optimizer[6], step_size = 100, gamma = 0.1)

            # 128
            lr_scheduler_G_128_2 = optim.lr_scheduler.StepLR(self.optimizer[7], step_size = 100, gamma = 0.1)
            lr_scheduler_D_Moire_128 = optim.lr_scheduler.StepLR(self.optimizer[8], step_size = 100, gamma = 0.1)
            lr_scheduler_D_Clear_128 = optim.lr_scheduler.StepLR(self.optimizer[9], step_size = 100, gamma = 0.1)

            # 64
            lr_scheduler_G_64_1 = optim.lr_scheduler.StepLR(self.optimizer[10], step_size = 100, gamma = 0.1)
            lr_scheduler_G_64_2 = optim.lr_scheduler.StepLR(self.optimizer[11], step_size = 100, gamma = 0.1)
            lr_scheduler_D_Moire_64 = optim.lr_scheduler.StepLR(self.optimizer[12], step_size = 100, gamma = 0.1)
            lr_scheduler_D_Clear_64 = optim.lr_scheduler.StepLR(self.optimizer[13], step_size = 100, gamma = 0.1)

            scheduler = [lr_scheduler_Net_256, lr_scheduler_Net_128, lr_scheduler_Net_64, lr_scheduler_Net_Demoire_TMB,
                         lr_scheduler_G_256_2, lr_scheduler_D_Moire_256, lr_scheduler_D_Clear_256,
                         lr_scheduler_G_128_2, lr_scheduler_D_Moire_128, lr_scheduler_D_Clear_128,
                         lr_scheduler_G_64_1, lr_scheduler_G_64_2, lr_scheduler_D_Moire_64, lr_scheduler_D_Clear_64]

        return scheduler

    def forward(self, x, res=64):
        if self.model_name == "cycle":
            return self.model(x, res) 
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = self.setup_optimizer()
        self.scheduler = self.setup_scheduler()
        if self.scheduler is not None:
            if self.model_name == "cycle":
                return self.optimizer, self.scheduler
            return [self.optimizer],[self.scheduler]
        else:
            return self.optimizer

    def training_epoch_end(self, outputs):
        if self.scheduler is not None:
            self.scheduler.step()
        return None

    def get_input(self, batch):
        
        number = batch['number']
        if self.mode == 'original':
            dataset_mode = batch['mode'] if isinstance(batch['mode'], str) else batch['mode'][0]
            if self.training or dataset_mode == 'train':
                x = batch['real_moire']
                y = batch['natural']
            else:
                x = batch['in_img']
                y = batch['label']
            moire_pattern = x
        
        elif self.mode == 'use_synthetic_moire_image_only':
            dataset_mode = batch['mode'] if isinstance(batch['mode'], str) else batch['mode'][0]
            if dataset_mode == 'train':
                moire_pattern = batch['moire_pattern']
                natural       = batch['natural']
                real_moire    = batch['real_moire']

                if self.blending_method == 'unidemoire':
                    with torch.no_grad():
                        # MIB:
                        init_blending_result, _ = self.moire_blending_model.init_blend(natural, moire_pattern)
                        # TRN:
                        trn_result = init_blending_result * self.moire_blending_model.refine_net(init_blending_result)    
                        min_val = torch.min(trn_result).to(self.device)
                        max_val = torch.max(trn_result).to(self.device)
                        trn_result = (trn_result - min_val) / (max_val - min_val)  

                        x = trn_result
                        y = natural
                
                elif self.blending_method == 'moirespace':
                    multiply_result = torch.clip(moire_pattern * natural, 0.0, 1.0)
                    x = multiply_result
                    y = natural
                
                elif self.blending_method == 'shooting':
                    shooting_result, shooting_gt = self.moire_blending_model(natural, self.device)
                    shooting_result = shooting_result.float()
                    shooting_gt = shooting_gt.float()
                    x = shooting_result
                    y = shooting_gt
                    
                elif self.blending_method == 'undem':
                    # Select the generator.
                    c = random.randint(0, 3)
                    netG_mo    = self.moire_blending_model[c*2].to(self.device)
                    encoder_mo = self.moire_blending_model[c*2 + 1].to(self.device)
                    
                    with torch.no_grad():
                        if self.undem_inference_data == 'tip_real_moire':
                            real_mo_feat = encoder_mo(xin=real_moire)
                        elif self.undem_inference_data == 'moire_pattern':
                            real_mo_feat = encoder_mo(xin=moire_pattern)
                        fake_mo = netG_mo(reference=real_mo_feat, xin=natural)
                    x = fake_mo
                    y = natural

                elif self.blending_method == 'cycle':
                    if self.cycle_inference_data == 'tip_real_moire':
                        x = batch['real_moire']
                        y = batch['natural']
                    elif self.cycle_inference_data == 'moire_pattern':
                        x = batch['moire_pattern']
                        y = batch['natural']
                
            elif dataset_mode == 'test':     
                x = batch['in_img']
                y = batch['label']
                moire_pattern = None

        elif self.mode == 'COMBINE_ONLINE':
            dataset_mode = batch['mode'] if isinstance(batch['mode'], str) else batch['mode'][0]
            if dataset_mode == 'train':
                moire_pattern = batch['moire_pattern']
                natural       = batch['natural']
                real_moire    = batch['real_moire']

                if self.blending_method == 'unidemoire':
                    with torch.no_grad():
                        # MIB
                        init_blending_result, _ = self.moire_blending_model.init_blend(natural, moire_pattern)
                        # TRN
                        trn_result = init_blending_result * self.moire_blending_model.refine_net(init_blending_result)      
                        min_val = torch.min(trn_result)
                        max_val = torch.max(trn_result)
                        trn_result = (trn_result - min_val) / (max_val - min_val)  
                        final_result = trn_result                   
                        
                        x = [final_result, real_moire]
                        y = [natural,      natural]

                elif self.blending_method == 'shooting':
                    shooting_result, shooting_gt = self.moire_blending_model(natural, self.device)
                    shooting_result = shooting_result.float()
                    shooting_gt = shooting_gt.float()
                    x = [shooting_result, real_moire]
                    y = [shooting_gt,     natural]

                elif self.blending_method == 'moirespace':
                    multiply_result = torch.clip(moire_pattern * natural, 0.0, 1.0)
                    x = [multiply_result, real_moire]
                    y = [natural,         natural]                                        

            elif dataset_mode == 'test':     
                x = batch['in_img']
                y = batch['label']
                moire_pattern = None

        return x, y, number, moire_pattern

    def train_step(self, x, y):
        if self.model_name == "ESDNet":
            out_1, out_2, out_3 = self(x)
            loss = self.loss_fn(out_1, out_2, out_3, y)
            y_hat = out_1
            
        elif self.model_name == "MBCNN":
            moires = x
            clear1 = y
            output3, output2, output1 = self(moires)
            loss = self.loss_fn(output3, output2, output1, clear1) + torch.Tensor([1e-10]).to(self.device)   
            y_hat = output1
            
        elif self.model_name == "PMTNet":
            out1, y2, y3 = self(x)
            loss = self.loss_fn(out1, y2, y3, y, self.learning_rate)
            y_hat = out1
        
        return y, y_hat, loss

    def val_step(self, x, y):
        if self.model_name == "ESDNet":
            out_1, _, _ = self(x)
            y_hat = out_1
            
        elif self.model_name == "MBCNN":
            moires = x
            clear1 = y
            _, _, output1 = self(moires)
            y_hat = output1
            
        elif self.model_name == "PMTNet":
            out1, y2, y3 = self(x)
            y_hat = out1
            
        return y, y_hat

    def training_step(self, batch, batch_idx):
        x, y, number, _ = self.get_input(batch)
        
        if self.model_name is not "cycle":
            rounds = 2 if self.mode == "COMBINE_ONLINE" else 1
            loss = 0
            if rounds == 1:
                in_img = x
                label  = y
                label, y_hat, loss_step = self.train_step(in_img, label)
                loss += loss_step
            else:
                for i in range(rounds):
                    in_img = x[i]
                    label  = y[i]
                    if i == 0:
                        #* blending-moire
                        label, y_hat, loss_step = self.train_step(in_img, label)
                        loss += loss_step
                    else:
                        #* Real-moire
                        if (self.use_additional_clean_image_only is not True) and len(number) == 5:
                            loss += loss
                        else:
                            label, y_hat, loss_step = self.train_step(in_img, label)
                            loss += loss_step                    

            loss = loss / rounds
            lr = self.optimizer.param_groups[0]['lr']
            self.learning_rate = self.optimizer.param_groups[0]['lr']
            self.log('lr', lr, prog_bar=True, logger=False)
        else:
            #! Just for the CycleModel
            [
                optimizer_Net_Demoire_256, optimizer_Net_Demoire_128, optimizer_Net_Demoire_64, optimizer_Net_Demoire_TMB,
                optimizer_G_256_2, optimizer_D_Moire_256, optimizer_D_Clear_256, 
                optimizer_G_128_2, optimizer_D_Moire_128, optimizer_D_Clear_128,
                optimizer_G_64_1, optimizer_G_64_2, optimizer_D_Moire_64, optimizer_D_Clear_64
            ] = self.optimizers()

        return loss

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x, y, number, moire_pattern = self.get_input(batch)

        if not only_inputs:
            rounds = 2 if self.mode == "COMBINE_ONLINE" else 1
            if rounds == 1:
                y, y_hat = self.val_step(x, y)
                log["output"] = y_hat
                log["gt"] = y
                log["inputs"] = x
                log["moire_pattern"] = moire_pattern
            else:
                for i in range(rounds):
                    if self.mode == "COMBINE_ONLINE":
                        tags = 'synthetic-moire' if i == 0 else 'real-moire'
                        # tags = 'blending-moire' if i == 0 else 'mib'
                    elif self.mode == "use_synthetic_moire_image_only":
                        tags = 'synthetic-moire-only'
                    else:
                        tags = 'original'
                    in_img = x[i]
                    label  = y[i]
                    label, y_hat = self.val_step(in_img, label)
                    log["inputs_%s" % tags] = in_img  
                    log["output_%s" % tags] = y_hat
                    log["gt"] = label
        return log

    def validation_step(self, batch, batch_idx):
        x, y, number, _ = self.get_input(batch)
    
        cur_psnr = 0.0
        cur_ssim = 0.0
        cur_lpips = 0.0
        b, c, h, w = x.size()
        
        # pad image such that the resolution is a multiple of 32
        w_pad = (math.ceil(w/32)*32 - w) // 2
        h_pad = (math.ceil(h/32)*32 - h) // 2
        w_odd_pad = w_pad
        h_odd_pad = h_pad
        if w % 2 == 1:
            w_odd_pad += 1
        if h % 2 == 1:
            h_odd_pad += 1
        
        x = self.img_pad(x, w_pad=w_pad, h_pad=h_pad, w_odd_pad=w_odd_pad, h_odd_pad=h_odd_pad)
        
        st = time.time()
        label, y_hat = self.val_step(x, y)
        cur_time = time.time()-st
        if h_pad != 0:
            y_hat = y_hat[:, :, h_pad:-h_odd_pad, :]
        if w_pad != 0:
            y_hat = y_hat[:, :, :, w_pad:-w_odd_pad]
        
        cur_lpips, cur_psnr, cur_ssim = self.compute_metrics.compute(y_hat, label, self.device)
        
        self.log('PSNR', cur_psnr, prog_bar=True, logger=False)
        self.log('SSIM', cur_ssim, prog_bar=True, logger=False)
        self.log('LPIPS', cur_lpips, prog_bar=True, logger=False)

        return {'cur_lpips': cur_lpips, 'cur_psnr': cur_psnr, 'cur_ssim': cur_ssim}

    def test_step(self, batch, batch_idx):
        x, y, number, _ = self.get_input(batch)
        cur_psnr = 0.0
        cur_ssim = 0.0
        cur_lpips = 0.0
        b, c, h, w = x.size()
        
        # pad image such that the resolution is a multiple of 32
        w_pad = (math.ceil(w/32)*32 - w) // 2
        h_pad = (math.ceil(h/32)*32 - h) // 2
        w_odd_pad = w_pad
        h_odd_pad = h_pad
        if w % 2 == 1:
            w_odd_pad += 1
        if h % 2 == 1:
            h_odd_pad += 1
        
        x = self.img_pad(x, w_pad=w_pad, h_pad=h_pad, w_odd_pad=w_odd_pad, h_odd_pad=h_odd_pad)
        
        st = time.time()
        # print(x.shape, y.shape)
        label, y_hat = self.val_step(x, y)
        cur_time = time.time()-st
        if h_pad != 0:
            y_hat = y_hat[:, :, h_pad:-h_odd_pad, :]
        if w_pad != 0:
            y_hat = y_hat[:, :, :, w_pad:-w_odd_pad]

        if self.evaluation_metric:
            cur_lpips, cur_psnr, cur_ssim = self.compute_metrics.compute(y_hat, label, self.device)
        
        if self.save_img:
            out_save = y_hat.detach().cpu()
            log_path = self.logger.save_dir
            root = os.path.join(log_path, 'test_images')
            os.makedirs(root, exist_ok=True)
            torchvision.utils.save_image(out_save, root + '/' + 'test_%s' % number[0] + '.jpg')

        return {'cur_lpips': cur_lpips, 'cur_psnr': cur_psnr, 'cur_ssim': cur_ssim}

    def test_epoch_end(self, outputs):
        # Compute the average of the metrics over the epoch
        if isinstance(outputs[0]['cur_psnr'], torch.Tensor) is True:
            avg_lpips = np.mean([x['cur_lpips'] for x in outputs])
            avg_psnr = np.mean([x['cur_psnr'].cpu() for x in outputs])
            avg_ssim = np.mean([x['cur_ssim'].cpu() for x in outputs])
        else:
            avg_lpips = np.mean([x['cur_lpips'] for x in outputs])
            avg_psnr = np.mean([x['cur_psnr'] for x in outputs])
            avg_ssim = np.mean([x['cur_ssim'] for x in outputs])
                    
        # Log the average metrics
        self.log('Avg. PSNR ', avg_psnr)
        self.log('Avg. SSIM ', avg_ssim)
        self.log('Avg. LPIPS', avg_lpips)

        return None

    def validation_epoch_end(self, outputs):
        # Compute the average of the metrics over the epoch
        if self.mode == 'original':
            avg_lpips = np.mean([x['cur_lpips'] for x in outputs])
            avg_psnr = np.mean([x['cur_psnr'] for x in outputs])
            avg_ssim = np.mean([x['cur_ssim'] for x in outputs])
        else:
            avg_lpips = np.mean([x['cur_lpips'] for x in outputs])
            avg_psnr = np.mean([x['cur_psnr'].cpu() for x in outputs])
            avg_ssim = np.mean([x['cur_ssim'].cpu() for x in outputs])

        # Log the average metrics
        self.log('Avg. PSNR ', avg_psnr)
        self.log('Avg. SSIM ', avg_ssim)
        self.log('Avg. LPIPS', avg_lpips)
        
        return None

    def img_pad(self, x, w_pad, h_pad, w_odd_pad, h_odd_pad):
        '''
        Here the padding values are determined by the average r,g,b values across the training set
        in FHDMi dataset. For the evaluation on the UHDM, you can also try the commented lines where
        the mean values are calculated from UHDM training set, yielding similar performance.
        '''
        x1 = F.pad(x[:, 0:1, ...], (w_pad, w_odd_pad, h_pad, h_odd_pad), value=0.3827)
        x2 = F.pad(x[:, 1:2, ...], (w_pad, w_odd_pad, h_pad, h_odd_pad), value=0.4141)
        x3 = F.pad(x[:, 2:3, ...], (w_pad, w_odd_pad, h_pad, h_odd_pad), value=0.3912)
        # x1 = F.pad(x[:, 0:1, ...], (w_pad, w_odd_pad, h_pad, h_odd_pad), value = 0.5165)
        # x2 = F.pad(x[:, 1:2, ...], (w_pad, w_odd_pad, h_pad, h_odd_pad), value = 0.4952)
        # x3 = F.pad(x[:, 2:3, ...], (w_pad, w_odd_pad, h_pad, h_odd_pad), value = 0.4695)
        y = torch.cat([x1, x2, x3], dim=1)

        return y

