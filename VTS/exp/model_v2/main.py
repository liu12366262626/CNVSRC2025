import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import hydra
from omegaconf import DictConfig
from .model import ASR_Diffusion_Model  # 自定义模型类
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
import torchvision.transforms as transforms
import math
from .tokenizer import CharTokenizer
import torch.distributed as dist
# 是否调试
debug_mode = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

import socket

def is_port_in_use(port, host="127.0.0.1"):
    """检查端口是否被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

# 如果调试模式 且 是主进程，启动 debugpy
if debug_mode:
    try:
        import debugpy
        debugpy.listen(("127.0.0.1", 5678))
        print("⚠️ Debugger waiting for attachment at port 5678 (main process only)")
        debugpy.wait_for_client()
        print("✅ Debugger attached!")
    except:
        print('already use')

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

class Reduction(nn.Module):
    def __init__(self, reduction="mean"):
        super(Reduction, self).__init__()

        assert reduction in ["sum", "mean", "mean_batch"]
        self.reduction = reduction

    def forward(self, x, n_elt=None):
        # Reduction
        if self.reduction == "sum":
            x = x.sum()
        elif self.reduction == "mean" and n_elt == None:
            x = x.mean()
        elif self.reduction == "mean" and n_elt != None:
            x = x.sum() / n_elt
        elif self.reduction == "mean_batch":
            x = x.mean(dim=0).sum()

        return x

class CTCLoss(nn.Module):
    def __init__(
        self, blank=0, reduction="mean", zero_infinity=False, assert_shorter=True
    ):
        super(CTCLoss, self).__init__()

        # mean: Sum Frames + Mean Batch
        # sum: Sum Frames + Sum Batch
        # default: Mean Frames + Mean Batch
        assert reduction in ["mean", "sum", "default"]

        # Loss
        self.loss = nn.CTCLoss(
            blank=blank,
            reduction="mean" if reduction == "default" else "none",
            zero_infinity=zero_infinity,
        )

        # Reduction
        self.reduction = (
            nn.Identity() if reduction == "default" else Reduction(reduction)
        )

        # Params
        self.assert_shorter = assert_shorter

    def forward(self, targets, outputs):
        # Unpack Targets
        y, y_len = targets

        # Unpack Outputs
        logits, logits_len = outputs

        # 检查 logits
        if torch.isnan(logits).any():
            print(f"[NaN Check] logits has NaN at step ")
        if torch.isinf(logits).any():
            print(f"[NaN Check] logits has Inf at step ")

        # 检查长度
        if (y_len > logits_len).any():
            print(f"[NaN Check] y_len > logits_len at step ")
            print(f"logits_len: {logits_len}")
            print(f"y_len: {y_len}")

        # 检查 log_softmax 后
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        if torch.isnan(log_probs).any():
            print(f"[NaN Check] log_softmax(logits) has NaN at step ")
        if torch.isinf(log_probs).any():
            print(f"[NaN Check] log_softmax(logits) has Inf at step ")

        # Assert
        if self.assert_shorter:
            assert (
                y_len <= logits_len
            ).all(), "logits length shorter than label length: \nlogits_len \n{} \ny_len \n{}".format(
                logits_len, y_len
            )

        # Compute Loss
        loss = self.loss(
            log_probs=torch.nn.functional.log_softmax(logits, dim=-1).transpose(
                0, 1
            ),  # (T, B, V)
            targets=y,
            input_lengths=logits_len,
            target_lengths=y_len,
        )

        # Reduction
        loss = self.reduction(loss)

        return loss

class CTCBeamDecoder:
    def __init__(self, id2char, beam_width=10, blank_id=0, log_probs_input=False,
                 cutoff_top_n=None, cutoff_prob=1.0):
        """
        Args:
            id2char (dict): 词表字典，id -> 字符（如 {0: '<blank>', 1: '<unk>', 2: '渔', ...}）
            beam_width (int): beam search 宽度
            blank_id (int): blank 符号的索引
            log_probs_input (bool): 输入是否为对数概率
            cutoff_top_n (int): 每步保留的最大 token 数
            cutoff_prob (float): 累积概率阈值
        """
        self.id2char = id2char
        self.vocab_size = len(id2char)
        self.blank_id = blank_id
        self.beam_width = beam_width
        self.log_probs_input = log_probs_input
        self.cutoff_top_n = cutoff_top_n if cutoff_top_n is not None else self.vocab_size
        self.cutoff_prob = cutoff_prob

    def decode(self, probs, seq_lens=None):
        if not self.log_probs_input:
            probs = probs.log_softmax(dim=-1)

        probs = probs.cpu()
        batch_size, max_seq_len, num_labels = probs.size()
        if seq_lens is None:
            seq_lens = torch.full((batch_size,), max_seq_len, dtype=torch.int32)

        beam_results = torch.full((batch_size, self.beam_width, max_seq_len), self.blank_id, dtype=torch.int32)
        beam_scores = torch.zeros(batch_size, self.beam_width)
        out_lens = torch.zeros(batch_size, self.beam_width, dtype=torch.int32)
        timesteps = torch.zeros(batch_size, self.beam_width, max_seq_len, dtype=torch.int32)

        for b in range(batch_size):
            seq_len = seq_lens[b].item()
            seq_probs = probs[b, :seq_len]  # (T, V)

            beams = [([], 0.0, [])]

            for t, prob_t in enumerate(seq_probs):
                top_probs, top_indices = torch.topk(prob_t, self.cutoff_top_n)

                # Apply cutoff prob
                sorted_probs, sorted_indices = torch.sort(top_probs, descending=True)
                if self.log_probs_input:
                    cumulative_prob = sorted_probs.exp().cumsum(0)
                else:
                    cumulative_prob = sorted_probs.cumsum(0)
                mask = cumulative_prob <= self.cutoff_prob
                mask = mask | (torch.arange(len(mask)) == 0)  # 至少保留一个最大概率
                valid_indices = top_indices[mask]

                new_beams = []
                for prefix, score, ts in beams:
                    for idx in valid_indices:
                        idx = idx.item()
                        p = prob_t[idx].item()
                        new_prefix = prefix + [idx]
                        new_score = score + p
                        new_ts = ts + [t]
                        new_beams.append((new_prefix, new_score, new_ts))

                new_beams.sort(key=lambda x: x[1], reverse=True)
                beams = new_beams[:self.beam_width]

            for i, (prefix, score, ts) in enumerate(beams):
                cleaned = []
                cleaned_ts = []
                prev = None
                for c, t_step in zip(prefix, ts):
                    if c != prev and c != self.blank_id:
                        cleaned.append(c)
                        cleaned_ts.append(t_step)
                    prev = c
                beam_results[b, i, :len(cleaned)] = torch.tensor(cleaned, dtype=torch.int32)
                beam_scores[b, i] = score
                out_lens[b, i] = len(cleaned)
                timesteps[b, i, :len(cleaned)] = torch.tensor(cleaned_ts, dtype=torch.int32)

        return beam_results, beam_scores, timesteps, out_lens

    def decode_to_text(self, logits, logits_len):
        """
        将 beam_results 的 ID 序列转成文本（字符序列）
        """
        batch_size, num_augments = logits.shape[0], 1
        logP = logits.log_softmax(dim=-1)
        beam_results, beam_scores, timesteps, out_lens = self.decode(logP, logits_len)
        beam_scores = beam_scores.reshape(batch_size, num_augments, self.beam_width)[:, :, 0]
        beam_results = beam_results.reshape(batch_size, num_augments, self.beam_width, -1)[:, :, 0]
        out_lens = out_lens.reshape(batch_size, num_augments, self.beam_width)[:, :, 0]

        batch_pred_tokens = []
        for b in range(batch_size):
            pred_tokens = beam_results[b][0][:out_lens[b][0]].tolist()
            batch_pred_tokens.append(pred_tokens)

        return batch_pred_tokens


class LightningModel(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters()
        self.model = ASR_Diffusion_Model(cfg)
        self.loss = CTCLoss()
        self.tokenizer = CharTokenizer(self.cfg.input.tokenizer_path)
        self.decoder = CTCBeamDecoder(
            id2char=self.tokenizer.id2char,
            beam_width=self.cfg.model.decode.beamsize,
            blank_id=self. tokenizer.blank_id,  
            log_probs_input=False,
            cutoff_top_n=30,        # 每步保留 top 3 token
            cutoff_prob=0.99       # 每步累计概率达 99% 就不再扩展
        )


        # load checkpoint
        param_dict = dict(self.model.named_parameters())
        
        checkpoint = torch.load(self.cfg.input.pretrained_path)

        new_state_dict = {}
        for key, value in checkpoint['model_state_dict'].items():
            new_key = 'model.' + key
            if new_key in param_dict.keys() and value.size() == param_dict[new_key].size():
                new_state_dict[new_key] = value

        result = self.model.load_state_dict(new_state_dict, strict = False)
        print('show')



    def training_step(self, batch, batch_idx):

        mel, length = batch['audio'], batch['audio_len']
        B = mel.size(0)
        device = mel.device
        _dh = calc_diffusion_hyperparams(**self.cfg.model.diffusion, fast = False)
        Alpha_bar = _dh["Alpha_bar"]
        Alpha_bar = Alpha_bar.to(device)

        mel = mel.permute(0,2,1)
        diffusion_steps = torch.randint(350, size=(B, 1, 1)).to(self.device)  # randomly sample diffusion steps from 1~T
        z = torch.normal(0, 1, size=mel.shape).cuda()
        mel = torch.sqrt(Alpha_bar[diffusion_steps]) * mel + torch.sqrt(1 - Alpha_bar[diffusion_steps]) * z  # compute x_t from q(x_t|x_0)
        inputs = mel, length
        targets = batch['label'], batch['label_len']

        output = self.model(inputs, diffusion_steps, targets)

        loss = self.loss(targets, output)
        # log loss & lr
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size= B, sync_dist= True)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", lr, on_step=True, logger=True)
        return loss

    def on_validation_epoch_start(self):
        self.val_outputs = []

    def validation_step(self, batch, batch_idx):
        mel, length = batch['audio'], batch['audio_len']
        B = mel.size(0)
        mel = mel.permute(0,2,1)
        device = mel.device
        _dh = calc_diffusion_hyperparams(**self.cfg.model.diffusion, fast = False)
        Alpha_bar = _dh["Alpha_bar"]
        Alpha_bar = Alpha_bar.to(device)
        time_step = 20
        diffusion_steps = time_step * torch.ones(B, 1, 1).long().cuda()
        z = torch.normal(0, 1, size=mel.shape).cuda()
        mel = torch.sqrt(Alpha_bar[diffusion_steps]) * mel + torch.sqrt(1 - Alpha_bar[diffusion_steps]) * z  
        inputs = mel, length
        targets = batch['label'], batch['label_len']

        output = self.model(inputs, diffusion_steps, targets)
        loss = self.loss(targets, output)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=B, sync_dist= True)

        # 每 200 个 epoch 才执行 beam search + CER
        cer_errors, cer_total = 0, 0
        if self.current_epoch % self.cfg.model.decode.every_epoch == 0:
            logits, logits_len = output[0], output[1]
            for b in range(B):
                pred_tokens = self.decoder.decode_to_text(logits, logits_len)
                pred_text = self.tokenizer.decode(pred_tokens[0])
                gt_text = self.tokenizer.decode(targets[0].tolist()[b])
                cer, error_token, total_token = self.compute_cer(pred_text, gt_text)
                cer_errors += error_token
                cer_total += total_token
            
            # logP = logits.log_softmax(dim=-1)
            # beam_results, beam_scores, timesteps, out_lens = self.decoder.decode(logP, logits_len)

            # beam_scores = beam_scores.reshape(B, 1, self.cfg.model.decode.beamsize)[:, :, 0]
            # beam_results = beam_results.reshape(B, 1, self.cfg.model.decode.beamsize, -1)[:, :, 0]
            # out_lens = out_lens.reshape(B, 1, self.cfg.model.decode.beamsize)[:, :, 0]

            # for b in range(B):
            #     pred_tokens = beam_results[b][0][:out_lens[b][0]].tolist()
            #     pred_text = self.tokenizer.decode(pred_tokens)
            #     gt_text = self.tokenizer.decode(targets[0].tolist()[b])
            #     cer, error_token, total_token = self.compute_cer(pred_text, gt_text)
            #     cer_errors += error_token
            #     cer_total += total_token

        self.val_outputs.append({
            "val_loss": loss.detach(),
            "cer_errors": cer_errors,
            "cer_total": cer_total
        })
        return loss

    def on_validation_epoch_end(self):
        total_errors = sum(x["cer_errors"] for x in self.val_outputs)
        total_tokens = sum(x["cer_total"] for x in self.val_outputs)

        if self.current_epoch % self.cfg.model.decode.every_epoch == 0:
            cer = total_errors / total_tokens if total_tokens > 0 else 0.0
            self.logger.experiment.add_scalar("CER/val_CER", cer, self.current_epoch)
            print(f"[Epoch {self.current_epoch}] val_CER: {cer:.4f} (errors: {total_errors}, total: {total_tokens})")

    def compute_cer(self, pred_text, gt_text):
        # 转成列表方便逐字比较
        pred_chars = list(pred_text)
        gt_chars = list(gt_text)

        # 长度对齐
        max_len = max(len(pred_chars), len(gt_chars))

        # 填充空字符，使长度一致（简单处理：缺失的补空）
        pred_chars += [''] * (max_len - len(pred_chars))
        gt_chars += [''] * (max_len - len(gt_chars))

        # 统计错字数量
        errors = sum(p != g for p, g in zip(pred_chars, gt_chars))

        # 计算 CER
        cer = errors / max_len if max_len > 0 else 0.0

        return cer, errors, max_len

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


    strategy = DDPStrategy(find_unused_parameters=False)

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
        # val_check_interval= 10,#step(update step) to check if （validate and save）
        check_val_every_n_epoch=1,
        logger= tensorboard_logger,
        num_sanity_val_steps = 0,
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

if __name__ == "__main__":
    main()
