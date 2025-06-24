import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import SequentialSampler, DataLoader
from tqdm import tqdm
import json
from hifi_gan.env import AttrDict
from hifi_gan.generator import Generator as Vocoder
from hifi_gan import utils as vocoder_utils
import soundfile as sf
import subprocess
from tqdm import tqdm
from pystoi.stoi import stoi
import Levenshtein
from mouthroi_processing.pipelines.model import AVSR
import csv
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torchvision
from model_v2.tokenizer import CharTokenizer
# import debugpy
# debugpy.listen(('127.0.0.1', 5678))
# print("⚠️ Debugger waiting for attachment at port 5678 (main process only)")
# debugpy.wait_for_client()
# print("Debugger connected!")



def metric(gt_text, hypothesis):


    # 移除空格，因为CER是基于字符计算的
    reference = gt_text.replace(" ", "")
    hypothesis = hypothesis.replace(" ", "")
    
    # 计算编辑距离
    distance = Levenshtein.distance(reference, hypothesis)
    
    # 计算总的字符数
    total_tokens = len(reference)
    
    # 计算CER
    cer = distance / total_tokens
    
    return {'error_token': distance, 'total_token': total_tokens, 'cer': cer}



def load_files(csv_path):
    prefix = '/home/liuzehua/task/VSR/data/preprocess'
    result = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            result.append([os.path.join(prefix, row[0], row[1]), row[3]])
    return result




@hydra.main(config_path="config", config_name="infer")
def main(cfg: DictConfig):
    
    tokenizer = CharTokenizer('/home/liuzehua/task/VTS/LipVoicer_revise/exp/model_v2/char_units.txt')
    csv_path = '/home/liuzehua/task/VTS/LipVoicer_revise/multi_test_char_2024.csv'
    infer_data = load_files(csv_path)
    print(len(infer_data))


    # load vsr model
    vsr_model = AVSR(cfg.vsr)
    vsr_model.load_checkpoint()
    vsr_model.to('cuda').eval()
    
    # preprocess method
    class FunctionalModule(torch.nn.Module):
        def __init__(self, functional):
            super().__init__()
            self.functional = functional

        def forward(self, input):
            return self.functional(input)


    video_pipeline = torch.nn.Sequential(
    FunctionalModule(lambda x: x / 255.0),
    torchvision.transforms.CenterCrop(88),
    torchvision.transforms.Grayscale(),
    torchvision.transforms.Normalize(0.421, 0.165),
    )

    total_error = 0
    total_token = 0
    dict_save = []
    for item in tqdm(infer_data, desc="Processing videos"):
        video_path, target = item
        
        video = torchvision.io.read_video(video_path, pts_unit='sec')[0] # T H W C
        video = video.permute(0, 3, 1, 2).contiguous().to('cuda')
        video = video_pipeline(video)

        pred_text = vsr_model.infer(video.cuda())

        gt_text = [int(x) for x in target.split()]
        gt_text = tokenizer.decode(gt_text)

        dict_ = metric(gt_text, pred_text)
        error , total , cer = dict_['error_token'], dict_['total_token'], dict_['cer']
        total_error = total_error + error
        total_token = total_token + total
        dict_save.append( {
            'video_path': video_path,
            'cer': cer,
            'target': gt_text,
            'pred': pred_text
        }
        )
        print(f'target: {gt_text}')
        print(f'pred: {pred_text}')
        print(f'cer: {cer}')

        print(f'current cer: {total_error/ total_token}')
    dict_save.append({
        'total_error': total_error,
        'total_token': total_token,
        'total_cer': total_error/ total_token
    })

    with open("/home/liuzehua/task/VTS/LipVoicer_revise/exp/inference/vsr_result1.json", "w") as f:
        json.dump(dict_save, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()



