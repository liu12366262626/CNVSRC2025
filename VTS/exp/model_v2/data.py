import logging
import os
import random
import numpy as np
import random
import torch
import csv
from . import utils as custom_utils
from torch.utils.data import DataLoader
import random
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from scipy.io.wavfile import read
from .dataloader.stft import TacotronSTFT, normalise_mel
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm
import math
from .tokenizer import CharTokenizer


logger = logging.getLogger(__name__)



class ASR_diffusion_Data(torch.utils.data.Dataset):
    def __init__(
            self,
            cfg,
            split
    ):
        # Initialize parameters
        self.label_path = os.path.join(cfg.data.label_dir, split) + '.csv'
        self.max_video_frame_size = cfg.data.max_video_frame_size
        self.min_video_frame_size = cfg.data.min_video_frame_size
        self.cfg = cfg
        self.dataset_name = split
        self.videos_window_size = cfg.data.videos_window_size
        self.tokenizer = CharTokenizer(self.cfg.input.tokenizer_path)
        # Load data
        paths, inds, tot, self.sizes, labels = self.load_data(self.label_path, self.max_video_frame_size, self.min_video_frame_size)
        self.data = []
        for i in range(len(paths)):
            self.data.append({
                'video_path': paths[i][0],
                'audio_path': paths[i][1],
                'video_frame': self.sizes[i],
                'label': labels[i],
            })

        logger.info("Checking for missing .melspec files in audio directory ...")
        for idx, item in enumerate(tqdm(self.data, desc="Checking & Generating", ncols=80), 1):
            audio_path = item['audio_path']
            mel_path = os.path.splitext(audio_path)[0] + '.melspec'

            if not os.path.exists(mel_path):
                logger.info(f"[{idx}/{len(self.data)}] Generating mel: {mel_path}")
                try:
                    audio, sr = torchaudio.load(audio_path)
                    if sr != self.cfg.data.sampling_rate:
                        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.cfg.data.sampling_rate)
                    audio = audio[0]  # mono
                    melspec = self.wav_to_mel(audio)
                    torch.save(melspec, mel_path)
                except Exception as e:
                    logger.warning(f"[{idx}/{len(self.data)}] Failed to generate mel for {audio_path}: {e}")

    def __len__(self):
        return len(self.data)

    def load_data(self, label_path, max_keep, min_keep):
        paths = []
        inds = []
        tot = 0
        sizes = []
        labels = []
        over_tot = 0
        below_tot = 0
        with open(label_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                video_path = row[0]
                audio_path = row[1]
                video_frame = int(row[2])
                audio_frame = int(row[3])
                label = row[4]
                if os.path.exists(video_path) and os.path.exists(audio_path):
                    if video_frame > max_keep:
                        over_tot = over_tot + 1
                        continue
                    if video_frame < min_keep:
                        below_tot = below_tot + 1
                        continue
                    inds.append(tot)
                    tot = tot + 1
                    paths.append((video_path, audio_path))
                    sizes.append(video_frame)
                    labels.append(label)
        logger.info(
            (
                f"max_keep={max_keep}, min_keep={min_keep}, "
                f"loaded {tot}, skipped {below_tot} short and {over_tot} long "
                f"longest-video-loaded={max(sizes)} frames, shortest-video-loaded={min(sizes)} frames"
            )
        )

        return paths, inds, tot, sizes, labels

    def align_audio_to_video(self, audio: np.ndarray, num_video_frames: int, sample_rate: int = 16000, fps: int = 25):
        """
        将音频对齐到给定的视频帧数（25fps）。
        audio: 一维 numpy 数组，音频采样信号
        num_video_frames: 视频帧数量
        sample_rate: 音频采样率（默认 16kHz）
        fps: 视频帧率（默认 25fps）
        
        返回对齐后的音频（长度为 num_video_frames * samples_per_frame）
        """
        samples_per_frame = sample_rate // fps  # 每帧对应多少个采样点
        target_length = num_video_frames * samples_per_frame

        if len(audio) > target_length:
            # 截断
            aligned_audio = audio[:target_length]
        elif len(audio) < target_length:
            # 补零
            padding = np.zeros(target_length - len(audio), dtype=audio.dtype)
            aligned_audio = np.concatenate([audio, padding])
        else:
            aligned_audio = audio  # 刚好对齐

        return aligned_audio
    
    def wav_to_mel(self, audio):
        stft = TacotronSTFT(filter_length=self.cfg.data.filter_length,
                            hop_length=self.cfg.data.hop_length,
                            win_length=self.cfg.data.win_length,
                            sampling_rate=self.cfg.data.sampling_rate,
                            mel_fmin=self.cfg.data.mel_fmin,
                            mel_fmax=self.cfg.data.mel_fmax)
        audio = audio / 1.1 / audio.abs().max()     # normalise max amplitude to be ~0.9
        audio = audio.unsqueeze(0)
        audio = torch.autograd.Variable(audio, requires_grad=False)
        melspec = stft.mel_spectrogram(audio)
        melspec = torch.squeeze(melspec, 0)

        return melspec  # [n_mels, time]

    def load_video(self, video_path):
        feats = custom_utils.load_video(video_path)
        feats = self.transform(feats)
        feats = np.expand_dims(feats, axis=-1)
        return feats

    def __getitem__(self, index):
        
        item = self.data[index]
        video_frame = item['video_frame']
        sampling_rate, audio_data = read(item['audio_path'])

                
        mel_path = os.path.splitext(item['audio_path'])[0] + '.melspec'
        melspec = torch.load(mel_path)

        # mouthroi shape: (T_mouth, 88, 88, 1)
        _, audio_len = melspec.shape
        expected_mel_len = video_frame * 4

        if audio_len > expected_mel_len:
            # 截断 mel 到合适长度
            melspec = melspec[:, :expected_mel_len]
        elif audio_len < expected_mel_len:
            # 在右侧 padding 0
            pad_len = expected_mel_len - audio_len
            melspec = F.pad(melspec, (0, pad_len))  # pad 右边

        audio_len = melspec.shape[1]
        

        min_val = math.log(1e-5)
        audio = ((melspec - min_val) / (-min_val / 2)) - 1
        audio = audio.permute(1,0)


        text = item['label']
        label = torch.LongTensor(self.tokenizer.encode(text))
        label_len = label.shape[0]

        audio_len = torch.tensor(audio_len, dtype=torch.long)
        label_len = torch.tensor(label_len, dtype=torch.long)

        return {'audio': audio, 'audio_len': audio_len, 'label': label, 'label_len': label_len}


    @staticmethod
    def collate_fn(batch):
        batch_audio = [sample['audio'] for sample in batch]
        batch_audio_len = [sample['audio_len'] for sample in batch]
        batch_label = [sample['label'] for sample in batch]
        batch_label_len = [sample['label_len'] for sample in batch]

        batch_audio = torch.nn.utils.rnn.pad_sequence(batch_audio, batch_first=True, padding_value=-1)
        batch_audio_len = torch.stack(batch_audio_len, axis=0)
        batch_label = torch.nn.utils.rnn.pad_sequence(batch_label, batch_first=True, padding_value=0)
        batch_label_len = torch.stack(batch_label_len, axis=0)

        
        return {'audio': batch_audio, 'audio_len': batch_audio_len, 'label': batch_label, 'label_len': batch_label_len}



def load_train(cfg, split):
    """
    加载训练数据的 DataLoader。

    Args:
        cfg: 配置字典，包含数据路径、批次大小等信息。

    Returns:
        DataLoader: 返回训练数据的 DataLoader。
    """
    train_dataset = ASR_diffusion_Data(cfg, split = split)  # 创建训练数据集

    # 检查是否在分布式环境中
    if torch.distributed.is_initialized():
        train_sampler = DistributedSampler(train_dataset, num_replicas=torch.distributed.get_world_size(), rank=torch.distributed.get_rank(), shuffle=True)
    else:
        train_sampler = RandomSampler(train_dataset) if cfg.data.shuffle else SequentialSampler(train_dataset)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.data.train_micro_batch_size_per_gpu,  # 使用配置中指定的批量大小
        sampler=train_sampler,  # 使用合适的 Sampler
        num_workers=cfg.data.num_workers,  # 可根据机器性能调整
        pin_memory=True,  # 加快数据传输到 GPU
        collate_fn=train_dataset.collate_fn  # 显式指定 collate_fn
    )
    return train_dataloader


def load_valid(cfg, split):
    """
    加载验证数据的 DataLoader。

    Args:
        cfg: 配置字典，包含数据路径、批次大小等信息。

    Returns:
        DataLoader: 返回验证数据的 DataLoader。
    """
    valid_dataset = ASR_diffusion_Data(cfg, split = split)  # 创建验证数据集

    # 检查是否在分布式环境中
    if torch.distributed.is_initialized():
        valid_sampler = DistributedSampler(valid_dataset, num_replicas=torch.distributed.get_world_size(), rank=torch.distributed.get_rank(), shuffle=False)
    else:
        valid_sampler = SequentialSampler(valid_dataset)

    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=1,  # 使用配置中指定的批量大小
        sampler=valid_sampler,  # 使用合适的 Sampler
        num_workers=cfg.data.num_workers,  # 可根据机器性能调整
        pin_memory=True,  # 加快数据传输到 GPU
        collate_fn=valid_dataset.collate_fn
    )
    return valid_dataloader