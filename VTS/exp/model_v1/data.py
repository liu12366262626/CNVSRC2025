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
import librosa
from librosa.util import normalize
import json
from scipy.io.wavfile import read
from .dataloader.stft import TacotronSTFT, normalise_mel
from .dataloader.video_reader import VideoReader
from PIL import Image, ImageEnhance
import cv2
import torchvision.transforms as transforms
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)

def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask



class AdaptiveTimeMask(torch.nn.Module):
    def __init__(self, window, stride):
        super().__init__()
        self.window = window
        self.stride = stride

    def forward(self, x):
        # x: [T, 88, 88]
        cloned = np.copy(x)
        length = cloned.shape[0]
        n_mask = int((length + self.stride - 0.1) // self.stride)
        ts = np.random.randint(0, self.window, size=(n_mask, 2))
        for t, t_end in ts:
            if length - t <= 0:
                continue
            t_start = random.randrange(0, length - t)
            if t_start == t_start + t:
                continue
            t_end += t_start
            if t_end > length:
                t_end = length
            cloned[t_start:t_end, :, :] = 0
        return cloned



class VTTS_Data(torch.utils.data.Dataset):
    def __init__(
            self,
            cfg,
            split
    ):
        # Initialize parameters
        self.label_path = os.path.join(cfg.data.label_dir, split) + '.csv'
        self.max_video_frame_size = cfg.data.max_video_frame_size
        self.min_video_frame_size = cfg.data.min_video_frame_size
        self.image_mean = cfg.data.image_mean
        self.image_std = cfg.data.image_std
        self.image_crop_size = cfg.data.image_crop_size
        self.image_aug = cfg.data.image_aug
        self.cfg = cfg
        self.dataset_name = split
        self.videos_window_size = cfg.data.videos_window_size

        if self.image_aug and 'train' in split:
            self.transform = custom_utils.Compose([
                custom_utils.Normalize(0.0, 255.0),
                custom_utils.RandomCrop((self.image_crop_size, self.image_crop_size)),
                custom_utils.HorizontalFlip(0.5),
                custom_utils.Normalize(self.image_mean, self.image_std)
            ])
        else:
            self.transform = custom_utils.Compose([
                custom_utils.Normalize(0.0, 255.0),
                custom_utils.CenterCrop((self.image_crop_size, self.image_crop_size)),
                custom_utils.Normalize(self.image_mean, self.image_std)
            ])

        self.face_image_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
        ])
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
        logger.info(f"Image transform: {self.transform}")

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
                if os.path.exists(video_path) and os.path.exists(audio_path) and os.path.exists(video_path.replace('video', 'origin_video')):
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

    @staticmethod
    def load_frame(clip_path):
        video_reader = VideoReader(clip_path, 1)
        start_pts, time_base, total_num_frames = video_reader._compute_video_stats()
        end_frame_index = total_num_frames - 1
        if end_frame_index < 0:
            clip = video_reader.read_video_only(start_pts, 1)
        else:
            clip = video_reader.read_video_only(random.randint(0, end_frame_index) * time_base, 1)
        frame = Image.fromarray(np.uint8(clip[0].to_rgb().to_ndarray())).convert('RGB')
        return frame

    @staticmethod
    def augment_image(image):
        if(random.random() < 0.5):
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.random()*0.6 + 0.7)
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(random.random()*0.6 + 0.7)
        return image

    def extract_window(self, mouthroi, mel):
        hop = self.cfg.data.hop_length

        # vid : T,C,H,W
        vid_2_aud = self.cfg.data.sampling_rate / self.cfg.data.fps / hop

        st_fr = random.randint(0, mouthroi.shape[0] - self.videos_window_size)
        mouthroi = mouthroi[st_fr:st_fr + self.videos_window_size]

        st_mel_fr = int(st_fr * vid_2_aud)
        mel_window_size = int(self.videos_window_size * vid_2_aud)

        mel = mel[:, st_mel_fr:st_mel_fr + mel_window_size]

        return mouthroi, mel
    
    def __getitem__(self, index):
        
        item = self.data[index]

        sampling_rate, audio_data = read(item['audio_path'])
        
        face_image = self.load_frame(item['video_path'].replace('video', 'origin_video'))

        mouthroi = self.load_video(item['video_path'])

                
        mel_path = os.path.splitext(item['audio_path'])[0] + '.melspec'
        melspec = torch.load(mel_path)

        # mouthroi shape: (T_mouth, 88, 88, 1)
        T_mouth = mouthroi.shape[0]
        T_mel = melspec.shape[1]
        expected_mel_len = T_mouth * 4

        if T_mel > expected_mel_len:
            # 截断 mel 到合适长度
            melspec = melspec[:, :expected_mel_len]
        elif T_mel < expected_mel_len:
            # 在右侧 padding 0
            pad_len = expected_mel_len - T_mel
            melspec = F.pad(melspec, (0, pad_len))  # pad 右边

        if self.dataset_name == 'train':

            mouthroi, melspec = self.extract_window(mouthroi, melspec)
            face_image = self.augment_image(face_image)
            mouthroi = torch.FloatTensor(mouthroi).squeeze(-1).unsqueeze(0) #(1, 25 , 88 , 88)
            face_image = self.face_image_transform(face_image)
            melspec = normalise_mel(melspec)
            return (melspec, mouthroi, face_image)
        
        else:

            audio, fs = torchaudio.load(item['audio_path'])
            text = item['label']

            # Normalisations & transforms
            audio = audio / 1.1 / audio.abs().max()
            mouthroi = torch.FloatTensor(mouthroi).squeeze(-1).unsqueeze(0) #(1, 25 , 88 , 88)
            face_image = self.face_image_transform(face_image)
            melspec = normalise_mel(melspec)
            return (melspec, audio, mouthroi, face_image, text, item['video_path'])


    @staticmethod
    def collate_fn(batch):
        return batch



def load_train(cfg, split):
    """
    加载训练数据的 DataLoader。

    Args:
        cfg: 配置字典，包含数据路径、批次大小等信息。

    Returns:
        DataLoader: 返回训练数据的 DataLoader。
    """
    train_dataset = VTTS_Data(cfg, split = split)  # 创建训练数据集

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
        pin_memory=True  # 加快数据传输到 GPU
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
    valid_dataset = VTTS_Data(cfg, split = split)  # 创建验证数据集

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
        pin_memory=True  # 加快数据传输到 GPU
    )
    return valid_dataloader