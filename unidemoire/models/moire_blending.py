import torch
import torch.nn as nn
import pytorch_lightning as pl
# from torch.nn.parameter import Parameter
from torch.optim import lr_scheduler
import torch.optim as optim
from omegaconf import OmegaConf
import glob

from .MIB.Blending import Blending
from .TRN.model import Uformer

from .utils.loss_util import *
from .utils.common import *

torch.autograd.set_detect_anomaly(True)

class MoireBlending_Model(pl.LightningModule):
    def __init__(self, model_name, network_config, loss_config=None, optimizer_config=None, ckpt_path=None, ignore_keys=[]):
        super().__init__()
        self.model_name            = model_name
        self.network_config        = network_config
        self.loss_config           = loss_config
        self.optimizer_config      = optimizer_config
        self.init_blending_args    = network_config["init_blending_args"]
        self.blending_network_args = network_config["blending_network_args"]

        # model
        self.model   = self.build_up_models()
        self.loss_fn = self.loss_function()

        if self.model_name == "UniDemoire":
            self.init_blend, self.refine_net = self.model        
        if ckpt_path is not None:
            print(f"Loading Checkpoint from {ckpt_path}")
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)


    def on_load_checkpoint(self, checkpoint): 
        print("Loading checkpoint...") 


    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        # only for very first batch
        if self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0:
            self.moire_image_encoder = None
    
    def get_config_from_ckpt_path(self, ckpt_path):
        if os.path.isfile(ckpt_path):              
            # paths = opt.resume.split("/")
            try:
                logdir = '/'.join(ckpt_path.split('/')[:-1])       
                print(f'Encoder dir is {logdir}')
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
    
    def val_mode(self, model):
        model = model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model
        
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        # print(f"Restored from {path}, self.mib_weight = {self.mib_weight}, self.init_blend.weight = {self.init_blend.weight}", sd['init_blend.weight'])
        
    def build_up_models(self):
        if self.model_name == "UniDemoire":
            init_blend = Blending(self.init_blending_args)
            refine_net = Uformer(
                    embed_dim=self.blending_network_args['embed_dim'],
                    depths=self.blending_network_args['depths'],
                    win_size=self.blending_network_args['win_size'],
                    modulator=self.blending_network_args['modulator'], 
                    shift_flag=self.blending_network_args['shift_flag']
            )    
            model = [init_blend, refine_net]
        else:
            model = None
        return model

    def loss_function(self):
        if self.model_name == "UniDemoire":
            Perceptual_Loss = PerceptualLoss()      
            TV_Loss = TVLoss()      
            ColorHistogram_Loss = ColorHistogramMatchingLoss()
            loss_fn = [Perceptual_Loss, TV_Loss, ColorHistogram_Loss]
        else:
            loss_fn = []

        return loss_fn

    def setup_optimizer(self):
        if self.model_name == "UniDemoire":
            optimizer = optim.Adam(
               [{
                   'params':
                        list(self.model[1].parameters()),         # self.refine_net
                   'initial_lr': 
                       self.learning_rate,
                    'lr': self.learning_rate         
                }], 
                betas=(self.optimizer_config["beta1"], self.optimizer_config["beta2"]) 
            )                         
        else:
            optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate)
        
        return optimizer

    def setup_scheduler(self):
        if self.model_name == "UniDemoire":
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, 
                T_0=self.optimizer_config["T_0"], 
                T_mult=self.optimizer_config["T_mult"], 
                eta_min=self.optimizer_config["eta_min"],
            )
        else:
            scheduler = None

        return scheduler

    def configure_optimizers(self):
        self.optimizer = self.setup_optimizer()
        self.scheduler = self.setup_scheduler()
        if self.scheduler is not None:
            return [self.optimizer],[self.scheduler]
        else:
            return self.optimizer

    def training_epoch_end(self, outputs):
        if self.scheduler is not None:
            self.scheduler.step()

    def get_input(self, batch):
        moire_pattern = batch['moire_pattern']
        natural       = batch['natural']
        real_moire    = batch['real_moire']
        number = batch['number']        
        
        return moire_pattern, natural, real_moire, number       

    def forward(self, moire_pattern, natural, real_moire):
        if self.model_name == "UniDemoire":
            moire_pattern = moire_pattern.to(self.device)
            natural       = natural.to(self.device)
            real_moire    = real_moire.to(self.device)
            self.init_blend.to(self.device)

            ##* Here's the MIB:
            mib_result, weight = self.init_blend(natural, moire_pattern)
            mib_result = mib_result.to(self.device)        
            self.log('w_mib', weight, prog_bar=True, logger=True)
                        
            ##* And here's the TRN:
            refine_result = mib_result * self.refine_net(mib_result, real_moire)
            min_val = torch.min(refine_result)
            max_val = torch.max(refine_result)
            refine_result = (refine_result - min_val) / (max_val - min_val)    
            refine_result = refine_result

            return mib_result, refine_result
        else: 
            return None

    def training_step(self, batch, batch_idx):
        if self.model_name == "UniDemoire":
            # Get data
            moire_pattern, natural, real_moire, number = self.get_input(batch)
            # Get Loss function
            Perceptual_Loss, TV_Loss, ColorHistogram_Loss = self.loss_fn
            #* Get the result
            mib_result, refine_result = self(moire_pattern, natural, real_moire)
            
            #* Calculate the losses
            content_loss = Perceptual_Loss(input=refine_result, target=mib_result, device=self.device, feature_layers=[0,1,2])
            color_loss = ColorHistogram_Loss(x=refine_result, y=real_moire, device=self.device)
            tv_loss = TV_Loss(refine_result)

            #** Total Loss:
            loss = color_loss + content_loss + 0.1 * tv_loss

            # Logging
            self.log('L_p', content_loss, prog_bar=True, logger=True)
            self.log('L_c', color_loss, prog_bar=True, logger=True)
            self.log('L_tv', tv_loss, prog_bar=True, logger=True)
            self.log('L_total', loss, prog_bar=False, logger=True)
            
        lr = self.optimizer.param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True, logger=False)
        
        return loss

    def feature_norm(self, feature):
        normed_feature = feature / feature.norm(dim=-1, keepdim=True)
        return normed_feature

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        moire_pattern, natural, real_moire, number = self.get_input(batch)
        log["natural"]       = natural
        log["moire_pattern"] = moire_pattern
        log["real_moire"]    = real_moire
        if not only_inputs:
            mib_result, trn_result = self(moire_pattern, natural, real_moire)
            log["init_blending_result"] = mib_result
            log["fusion_result"]        = trn_result

        return log