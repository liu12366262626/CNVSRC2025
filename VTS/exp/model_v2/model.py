
import torch.nn as nn
import logging

import torch.nn as nn

import torch.nn.functional as F
import torch
import numpy as np
from .models import networks


logger = logging.getLogger(__name__)

def swish(x):
    return x * torch.sigmoid(x)

def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    """
    Embed a diffusion step $t$ into a higher dimensional space
    E.g. the embedding vector in the 128-dimensional space is
    [sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)), cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]

    Parameters:
    diffusion_steps (torch.long tensor, shape=(batchsize, 1)):
                                diffusion steps for batch data
    diffusion_step_embed_dim_in (int, default=128):
                                dimensionality of the embedding space for discrete diffusion steps

    Returns:
    the embedding vectors (torch.tensor, shape=(batchsize, diffusion_step_embed_dim_in)):
    """

    assert diffusion_step_embed_dim_in % 2 == 0

    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)
    _embed = torch.exp(torch.arange(half_dim) * -_embed).cuda()
    _embed = diffusion_steps * _embed
    diffusion_step_embed = torch.cat((torch.sin(_embed),
                                      torch.cos(_embed)), 1)

    return diffusion_step_embed


class AudioEfficientConformerInterCTC(nn.Module):

    def __init__(self, vocab_size=4469, att_type="patch", interctc_blocks=[3, 6, 10, 13], T=400, beta_0=0.0001, beta_T=0.02):
        super(AudioEfficientConformerInterCTC, self).__init__()
        # Actually we do not need Interctc here, we only use conformer
        self.encoder = networks.AudioEfficientConformerEncoder(vocab_size=vocab_size, att_type=att_type, interctc_blocks=interctc_blocks)

        # the shared two fc layers for diffusion step embedding
        self.diffusion_step_embed_dim_in = 128
        self.diffusion_step_embed_dim_mid = 512
        self.diffusion_step_embed_dim_out = 512
        self.fc_t1 = nn.Linear(self.diffusion_step_embed_dim_in, self.diffusion_step_embed_dim_mid)
        self.fc_t2 = nn.Linear(self.diffusion_step_embed_dim_mid, self.diffusion_step_embed_dim_out)

    def forward(self, inputs, diffusion_steps):
        # Embed diffusion step
        diffusion_steps = diffusion_steps.view(inputs[0].size(0), 1)
        diffusion_step_embed = calc_diffusion_step_embedding(diffusion_steps, self.diffusion_step_embed_dim_in)
        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))
        
        x, lengths = inputs
        x, lengths, interctc_outputs = self.encoder(x, lengths, diffusion_step_embed)
        outputs = {"outputs": [x, lengths]}
        outputs.update(interctc_outputs)
        return outputs



class ASR_Diffusion_Model(nn.Module):  # 继承 PyTorch 的 nn.Module
    def __init__(self, cfg, mode = 'train'):
        super(ASR_Diffusion_Model, self).__init__()
        self.cfg = cfg
        # 读取文件并统计行数
        self.vocab_size = 0
        with open(cfg.input.tokenizer_path, 'r', encoding='utf-8') as f:
            for _ in f:
                self.vocab_size += 1

        self.model = AudioEfficientConformerInterCTC(vocab_size=self.vocab_size, att_type=cfg.model.asr.att_type, interctc_blocks=cfg.model.asr.interctc_blocks)

    def prepare_deepspeed(self):
        """为 DeepSpeed 提供模型参数"""       
        return self.parameters()
    
    

    def forward(self, inputs, diffusion_steps, target):
        output = self.model(inputs, diffusion_steps)
        return output['outputs']


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


