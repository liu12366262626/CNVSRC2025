import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import hydra
from omegaconf import DictConfig
from .model import VTTS_Model  # 自定义模型类
from .data import load_train, load_valid  # 自定义数据集类
import debugpy
import torch
import torch.nn as nn
import lightning as L
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import io
from PIL import Image
import torchvision.transforms as transforms
import math
# 是否调试
debug_mode = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
if debug_mode:
    debugpy.listen(('127.0.0.1', 5678))
    print("⚠️ Debugger waiting for attachment at port 5678 (main process only)")
    debugpy.wait_for_client()
    print("Debugger connected!")


def calc_diffusion_hyperparams(T, beta_0, beta_T, beta=None, fast=False):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value,
                                where any beta_t in the middle is linearly interpolated

    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
        These cpu tensors are changed to cuda tensors on each individual gpu
    """

    if fast and beta is not None:
        Beta = torch.tensor(beta)
        T = len(beta)
    else:
        Beta = torch.linspace(beta_0, beta_T, T)
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t - 1]  # \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
        Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (
            1 - Alpha_bar[t]
        )  # \tilde{\beta}_t = \beta_t * (1-\bar{\alpha}_{t-1}) / (1-\bar{\alpha}_t)
    Sigma = torch.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t

    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = (
        T,
        Beta.cuda(),
        Alpha.cuda(),
        Alpha_bar.cuda(),
        Sigma,
    )
    return _dh

def denormalise_mel(melspec, min_val=math.log(1e-5)): 
    melspec = ((melspec + 1) * (-min_val / 2)) + min_val
    return melspec

class LightningModel(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.model = VTTS_Model(cfg)
        self.loss = nn.L1Loss()


    def training_step(self, batch, batch_idx):
        
        melspec, mouthroi, face_image = batch




        _dh = calc_diffusion_hyperparams(**self.cfg.model.diffusion, fast = False)
        T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]

        B, C, L = melspec.shape  # B is batchsize, C=80, L is number of melspec frames
        diffusion_steps = torch.randint(T, size=(B,1,1)).cuda()  # randomly sample diffusion steps from 1~T
        z = torch.normal(0, 1, size=melspec.shape).cuda()
        transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * melspec + torch.sqrt(1-Alpha_bar[diffusion_steps]) * z  # compute x_t from q(x_t|x_0)
        cond_drop_prob = self.cfg.model.melgen.cond_drop_prob
        epsilon_theta = self.model(transformed_X, mouthroi, face_image, diffusion_steps.view(B,1), cond_drop_prob)

        loss = self.loss(epsilon_theta, z)

        # log loss & lr
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size= B)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", lr, on_step=True, logger=True)
        return loss


    def validation_step(self, batch, batch_idx):
        melspec, audio, mouthroi, face_image, text, video_path = batch
        _dh = calc_diffusion_hyperparams(**self.cfg.model.diffusion, fast=True)
        if batch_idx < 5:
            generate_mel = self.model.generate(
            diffusion_hyperparams = _dh,
            w_video = 2,
            condition=(mouthroi, face_image),
            )
            groundtruth_melspec = denormalise_mel(melspec)

            # 只取第一个样本
            gen = generate_mel[0].detach().cpu().numpy()
            gt = groundtruth_melspec[0].detach().cpu().numpy()

            # 绘图
            fig, axs = plt.subplots(2, 1, figsize=(10, 8))
            axs[0].imshow(gt, origin='lower', aspect='auto', interpolation='none')
            axs[0].set_title("Ground Truth Mel")

            axs[1].imshow(gen, origin='lower', aspect='auto', interpolation='none')
            axs[1].set_title("Generated Mel")

            # 转成 TensorBoard 图像
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image = Image.open(buf)
            image = transforms.ToTensor()(image)

            # 记录到 TensorBoard
            self.logger.experiment.add_image(
                f"mel_compare/batch{batch_idx}",
                image,
                global_step=self.global_step
            )

            plt.close(fig)


        melspec = melspec.to(self.device)
        mouthroi = mouthroi.to(self.device)
        face_image = face_image.to(self.device)

        

        B, C, L = melspec.shape       # B = 1
        T = _dh["T"]
        num_samples = 20              # 评估 20 个不同 diffusion step

        # Repeat inputs 20 次
        melspec = melspec.repeat(num_samples, 1, 1)        # [20, 80, L]
        mouthroi = mouthroi.repeat(num_samples, 1, 1, 1, 1)  # [20, 1, T, 88, 88]
        face_image = face_image.repeat(num_samples, 1, 1, 1)  # [20, 3, 224, 224]

        # 随机生成 20 个 diffusion step
        diffusion_steps = torch.randint(T, size=(num_samples, 1), device=self.device)
        z = torch.randn_like(melspec)

        alpha_bar_t = _dh["Alpha_bar"][diffusion_steps].view(num_samples, 1, 1)  # [20, 1, 1]

        x_t = torch.sqrt(alpha_bar_t) * melspec + torch.sqrt(1 - alpha_bar_t) * z

        epsilon_theta = self.model(
            x_t, mouthroi, face_image,
            diffusion_steps,  # [20, 1]
            cond_drop_prob=self.cfg.model.melgen.cond_drop_prob
        )

        loss = self.loss(epsilon_theta, z)  # 默认为 element-wise mean

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size= B)


        return loss




    def configure_optimizers(self):
        # 自定义的 tri_stage 学习率调度器
        def tri_stage_scheduler(optimizer, warmup_steps, hold_steps, decay_steps, final_lr_scale, total_steps):
            def lr_lambda(step: int):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                elif step < warmup_steps + hold_steps:
                    return 1.0
                elif step < total_steps:
                    return max(final_lr_scale, 
                               1 - (float(step - warmup_steps - hold_steps) / float(max(1, decay_steps))))
                else:
                    return final_lr_scale
            return LambdaLR(optimizer, lr_lambda)

        # 获取优化器的配置
        lr = self.cfg.training.optimization.lr
        beta1 = self.cfg.training.optimizer.beta1
        beta2 = self.cfg.training.optimizer.beta2
        eps = self.cfg.training.optimizer.adam_eps
        max_update = self.cfg.training.optimization.max_update
        warmup_steps = self.cfg.training.lr_scheduler.warmup_steps
        hold_steps = self.cfg.training.lr_scheduler.hold_steps
        decay_steps = self.cfg.training.lr_scheduler.decay_steps
        final_lr_scale = self.cfg.training.lr_scheduler.final_lr_scale

        # 创建 Adam 优化器
        optimizer = Adam(self.parameters(), lr=lr, betas=(beta1, beta2), eps=eps)

        # 创建学习率调度器
        scheduler = tri_stage_scheduler(
            optimizer,
            warmup_steps=warmup_steps,
            hold_steps=hold_steps,
            decay_steps=decay_steps,
            final_lr_scale=final_lr_scale,
            total_steps=max_update
        )

        return [optimizer], [{
            'scheduler': scheduler,
            'interval': 'step',  # 在每个 step 更新
            'frequency': 1
        }]



@hydra.main(config_path="/home/liuzehua/task/VSR/VSP-LLM/exp_2/model_v4/config", config_name="train")
def main(cfg: DictConfig):
    # 设置数据加载器
    train_dataloader = load_train(cfg, cfg.data.train_file)
    
    valid_dataloader = load_valid(cfg, cfg.data.valid_file)

    model = LightningModel(cfg)


    strategy = DDPStrategy(find_unused_parameters=True)

    checkpoint_trainloss_callback = ModelCheckpoint(
        monitor='train_loss',
        dirpath=cfg.save.save_checkpoint,
        filename='{step}_{train_loss:.4f}',  # 用 _ 替代 =
        save_top_k=4,
        mode='min',
        save_last= True
    )


    checkpoint_loss_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=cfg.save.save_checkpoint,
        filename='{step}_{val_loss:.4f}',  # 用 _ 替代 =
        save_top_k=2,
        mode='min'
    )


    # 设置 TensorBoard logger
    tensorboard_logger = TensorBoardLogger(
        save_dir=cfg.save.save_tensorboard,  # 日志保存路径
        name='tensorboard_logs'  # 日志文件夹名
    )
    # 初始化 Trainer
    trainer = L.Trainer(
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        max_steps=cfg.training.max_steps,
        accelerator="gpu",
        devices=cfg.input.num_gpus,
        log_every_n_steps=10,
        strategy=strategy,
        callbacks=[checkpoint_loss_callback, checkpoint_trainloss_callback],
        val_check_interval= 1000,#step(update step) to check if （validate and save）
        check_val_every_n_epoch=None,
        logger= tensorboard_logger,
        num_sanity_val_steps = 0
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

if __name__ == "__main__":
    main()
