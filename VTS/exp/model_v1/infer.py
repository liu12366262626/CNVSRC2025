import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from model import VTTS_Model
from main import calc_diffusion_hyperparams
from data import  VTTS_Data
import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import SequentialSampler, DataLoader

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_v2.model import ASR_Diffusion_Model

import debugpy
debugpy.listen(('127.0.0.1', 5678))
print("⚠️ Debugger waiting for attachment at port 5678 (main process only)")
debugpy.wait_for_client()
print("Debugger connected!")






def sampling(vtts_model, asr_guidance_net,
             diffusion_hyperparams, tokenizer
             w_video, w_asr, asr_start,
             guidance_text,
             condition):
    
    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T

    # tokenize text
    if asr_guidance_net is not None:
        text_tokens = torch.LongTensor(tokenizer.encode(guidance_text))
        text_tokens = text_tokens.unsqueeze(0).cuda()


    print('show')
    








@hydra.main(config_path="config", config_name="infer")
def main(cfg: DictConfig):

    # load vtts model
    checkpoint = torch.load(cfg.vtts_path)
    config = checkpoint['hyper_parameters']['cfg']
    vtts_model = VTTS_Model(config)
    new_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        new_key = key[6:]
        new_state_dict[new_key] = value
    vtts_model.load_state_dict(new_state_dict, strict= False)
    vtts_model.eval()
    diffusion_hyperparams = calc_diffusion_hyperparams(**config.model.diffusion, fast = True)


    # load dataset
    dataset = VTTS_Data(config, cfg.split)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,  # 使用配置中指定的批量大小
        sampler=sampler,  # 使用合适的 Sampler
        num_workers=2,  # 可根据机器性能调整
    )

    # load asr_guidance model 
    checkpoint = torch.load(cfg.asr_guidance_path)
    config = checkpoint['hyper_parameters']['cfg']
    asr_guidance_model = ASR_Diffusion_Model(config)
    new_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        new_key = key[6:]
        new_state_dict[new_key] = value
    asr_guidance_model.load_state_dict(new_state_dict, strict= False)
    asr_guidance_model.eval()

    
    for batch in dataloader:
        melspec, audio, mouthroi, face_image, text, video_path = batch
        
        sampling(vtts_model, asr_guidance_model, diffusion_hyperparams, 
                 w_video = 2, w_asr = 1.5, asr_start = 270,
                guidance_text = text, condition = (mouthroi.cuda(), face_image.cuda()))



if __name__ == '__main__':
    main()



