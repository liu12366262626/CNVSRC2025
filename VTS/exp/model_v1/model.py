
import torch.nn as nn
import logging
import os
import torch.nn as nn
import json
import torch.nn.functional as F
import torch
import torchvision
from .models.face_model import Resnet18
from .models.lipreading_model import Lipreading
import os
import sys


from .models.wavenet import WaveNet


logger = logging.getLogger(__name__)


# builder for facial attributes analysis stream
def build_facial( pool_type='maxpool', fc_out=512, with_fc=False):
    pretrained = False
    original_resnet = torchvision.models.resnet18(pretrained)
    net = Resnet18(original_resnet, pool_type=pool_type, with_fc=with_fc, fc_in=512, fc_out=fc_out)
    return net

#builder for lipreading stream 
def build_lipreadingnet(cfg):
    print('Lipreading configuration file loaded.')
    tcn_options = { 'num_layers': cfg['tcn_num_layers'],
                    'kernel_size': list(cfg['tcn_kernel_size']),
                    'dropout': cfg['tcn_dropout'],
                    'dwpw': cfg['tcn_dwpw'],
                    'width_mult': cfg['tcn_width_mult']}
    net = Lipreading(tcn_options=tcn_options,
                    backbone_type=cfg['backbone_type'],
                    relu_type=cfg['relu_type'],
                    width_mult=cfg['width_mult'],
                    extract_feats=True)

    return net

def build_diffwave_model( model_cfg):
    cond_feat_size = 640        # size of feature dimension for the conditioner
    model = WaveNet(cond_feat_size, **model_cfg)
    return model




class VTTS_Model(nn.Module):  # 继承 PyTorch 的 nn.Module
    def __init__(self, cfg, mode = 'train'):
        super(VTTS_Model, self).__init__()
        self.cfg = cfg

        self.net_facial =  build_facial(fc_out=128, with_fc=True)
        self.net_lipreading = build_lipreadingnet(cfg.model.lip_reading)
        self.net_diffwave = build_diffwave_model(cfg.model.melgen)


    
        # classifier guidance null conditioners
        torch.manual_seed(0)        # so we have the same null tokens on all nodes
        self.register_buffer("mouthroi_null", torch.randn(1, 1, 1, 88, 88))  # lips regions frames are 88x88 each
        self.register_buffer("face_null", torch.randn(1, 3, 224, 224))  # face image size is 224x224


    def prepare_deepspeed(self):
        """为 DeepSpeed 提供模型参数"""       
        return self.parameters()
    
    

    def forward(self, melspec, mouthroi, face_image, diffusion_steps, cond_drop_prob):
        # classifier guidance
        # melspec is noised ground_truth Melspec 
        batch = melspec.shape[0]
        if cond_drop_prob > 0:
            prob_keep_mask = self.prob_mask_like((batch, 1, 1, 1, 1), 1.0 - cond_drop_prob, melspec.device)
            _mouthroi = torch.where(prob_keep_mask, mouthroi, self.mouthroi_null)
            _face_image = torch.where(prob_keep_mask.squeeze(1), face_image, self.face_null)
        else:
            _mouthroi = mouthroi
            _face_image = face_image

        # pass through visual stream and extract lipreading features
        lipreading_feature = self.net_lipreading(_mouthroi)
        
        # pass through visual stream and extract identity features
        identity_feature = self.net_facial(_face_image)

        # what type of visual feature to use
        identity_feature = identity_feature.repeat(1, 1, 1, lipreading_feature.shape[-1])
        visual_feature = torch.cat((identity_feature, lipreading_feature), dim=1)
        visual_feature = visual_feature.squeeze(2)  # so dimensions are B, C, num_frames

        output = self.net_diffwave((melspec, diffusion_steps), cond=visual_feature)

        return output


    def generate(self, diffusion_hyperparams, w_video, condition):
        _dh = diffusion_hyperparams
        T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
        mouthroi, face_image = condition
        assert len(Alpha) == T
        assert len(Alpha_bar) == T
        assert len(Sigma) == T
        x = torch.normal(0, 1, size=(mouthroi.shape[0], 80, mouthroi.shape[2] * 4)).cuda()
        with torch.no_grad():
            for t in range(T - 1, -1, -1):
                diffusion_steps = (
                    t * torch.ones((x.shape[0], 1))
                ).cuda()  # use the corresponding reverse step
                epsilon_theta = self.forward(
                    x, mouthroi, face_image, diffusion_steps, cond_drop_prob=0
                )  # predict \epsilon according to \epsilon_\theta
                epsilon_theta_uncond = self.forward(
                    x, mouthroi, face_image, diffusion_steps, cond_drop_prob=1
                )
                epsilon_theta = (
                    1 + w_video
                ) * epsilon_theta - w_video * epsilon_theta_uncond

                x = (
                    x - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta
                ) / torch.sqrt(
                    Alpha[t]
                )  # update x_{t-1} to \mu_\theta(x_t)
                if t > 0:
                    x = (
                        x + Sigma[t] * torch.normal(0, 1, size=x.shape).cuda()
                    )  # add the variance term to x_{t-1}
        return x




    @staticmethod
    def prob_mask_like(shape, prob, device):
        if prob == 1:
            return torch.ones(shape, device=device, dtype=torch.bool)
        elif prob == 0:
            return torch.zeros(shape, device=device, dtype=torch.bool)
        else:
            return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob
