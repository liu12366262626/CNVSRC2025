import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
from pystoi.stoi import stoi
import Levenshtein
from mouthroi_processing.pipelines.model import AVSR

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_v2.model import ASR_Diffusion_Model
from model_v2.tokenizer import CharTokenizer
from model_v2.main import CTCBeamDecoder, CTCLoss
from model_v1.model import VTTS_Model
from model_v1.main import calc_diffusion_hyperparams
from model_v1.data import  VTTS_Data
from model_v1.dataloader.stft import denormalise_mel
# import debugpy
# debugpy.listen(('127.0.0.1', 5678))
# print("⚠️ Debugger waiting for attachment at port 5678 (main process only)")
# debugpy.wait_for_client()
# print("Debugger connected!")


# def metric(ref_wav, gen_wav, sample_rate, gt_text):

#     stoi = stoi(ref_wav, gen_wav, sample_rate, extended=False)



#     # 移除空格，因为CER是基于字符计算的
#     reference = gt_text.replace(" ", "")
#     hypothesis = hypothesis.replace(" ", "")
    
#     # 计算编辑距离
#     distance = Levenshtein.distance(reference, hypothesis)
    
#     # 计算总的字符数
#     total_tokens = len(reference)
    
#     # 计算CER
#     cer = distance / total_tokens
    
#     return {'stoi': stoi, 'error_token': distance, 'total_token': total_tokens}




def sampling(vtts_model, asr_guidance_net,
             diffusion_hyperparams, tokenizer,
             w_video, w_asr, asr_start,
             guidance_text,
             condition):
        
    decoder = CTCBeamDecoder(
                id2char=tokenizer.id2char,
                beam_width=16,
                blank_id=tokenizer.blank_id,  
                log_probs_input=False,
                cutoff_top_n=30,        # 每步保留 top 3 token
                cutoff_prob=0.99       # 每步累计概率达 99% 就不再扩展
            )
    ctcloss = CTCLoss()

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T

    # tokenize text
    if asr_guidance_net is not None:
        text_tokens = torch.LongTensor(tokenizer.encode(guidance_text))
        text_tokens = text_tokens.unsqueeze(0).cuda()
        print(f'guidance text: {guidance_text}')
    
    mouthroi, face_image = condition
    x = torch.normal(0, 1, size=(mouthroi.shape[0], 80, mouthroi.shape[2]*4)).cuda()
    with torch.no_grad():
        for t in tqdm(range(T-1, -1, -1)):
            diffusion_steps = (t * torch.ones((x.shape[0], 1))).cuda()  # use the corresponding reverse step
            epsilon_theta = vtts_model(x, mouthroi, face_image, diffusion_steps, cond_drop_prob=0)   # predict \epsilon according to \epsilon_\theta
            epsilon_theta_uncond = vtts_model(x, mouthroi, face_image, diffusion_steps, cond_drop_prob=1)
            epsilon_theta = (1+w_video) * epsilon_theta - w_video * epsilon_theta_uncond

            if asr_guidance_net is not None and t <= asr_start:
                with torch.enable_grad():
                    length_input = torch.tensor([x.shape[2]]).cuda()
                    inputs = x.detach().requires_grad_(True), length_input
                    targets = text_tokens, torch.tensor([text_tokens.shape[1]]).cuda()
                    asr_guidance_net.device = torch.device("cuda")
                    # batch_losses = asr_guidance_net(inputs, diffusion_steps, targets, compute_metrics=True, verbose=0)[0]
                    loss = ctcloss(targets, asr_guidance_net(inputs, diffusion_steps, targets))
                    asr_grad = torch.autograd.grad(loss, inputs[0])[0]
                    asr_guidance_net.device = torch.device("cpu")
                grad_normaliser = torch.norm(epsilon_theta / torch.sqrt(1 - Alpha_bar[t])) / torch.norm(asr_grad)
                epsilon_theta = epsilon_theta + torch.sqrt(1 - Alpha_bar[t]) * w_asr * grad_normaliser * asr_grad

            x = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])  # update x_{t-1} to \mu_\theta(x_t)
            if t > 0:
                x = x + Sigma[t] * torch.normal(0, 1, size=x.shape).cuda()  # add the variance term to x_{t-1}


            if t % 20 == 0:
                if asr_guidance_net is not None and t <= asr_start:
                    inputs = x, length_input
                    outputs_ao = asr_guidance_net(inputs, diffusion_steps, targets)
                    preds_ao = decoder.decode_to_text(outputs_ao[0], outputs_ao[1])[0]
                    pred_text = tokenizer.decode(preds_ao)
                    print(pred_text)

    return x
    





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
    vtts_model.eval().to('cuda')
    diffusion_hyperparams = calc_diffusion_hyperparams(**config.model.diffusion, fast = True)


    # load dataset
    dataset = VTTS_Data(config, cfg.split)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,  
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
    asr_guidance_model.load_state_dict(new_state_dict, strict= True)
    asr_guidance_model.eval().to('cuda')
    tokenizer = CharTokenizer(config.input.tokenizer_path)


    # load vsr model
    vsr_model = AVSR(cfg.vsr)
    vsr_model.load_checkpoint()
    vsr_model.to('cuda').eval()
    
    # load vocoder
    config_file = os.path.join(os.path.dirname(__file__), 'hifi_gan', 'config.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    vocoder = Vocoder(h).cuda()
    checkpoint_file = os.path.join(os.path.dirname(__file__), 'hifi_gan', 'g_02400000')
    state_dict_g = vocoder_utils.load_checkpoint(checkpoint_file, 'cuda')
    vocoder.load_state_dict(state_dict_g['generator'])
    vocoder.eval()
    vocoder.remove_weight_norm()
    
    for batch in tqdm(dataloader, desc="Processing batches"):
        melspec, _, mouthroi, face_image, text, video_path = batch
        video_path = video_path[0]


        guidance_text = vsr_model.infer(mouthroi.squeeze(0).squeeze(0).unsqueeze(1).cuda())
        print(f'vsr result : {guidance_text}')
        melspec = sampling(vtts_model, asr_guidance_model, diffusion_hyperparams, tokenizer,
                 w_video = 2, w_asr = 1.5, asr_start = 270,
                guidance_text = guidance_text, condition = (mouthroi.cuda(), face_image.cuda()))


        melspec = denormalise_mel(melspec)

        print('show')
        os.makedirs(cfg.save_path, exist_ok=True)

        video_name = video_path.split('/')[-1].replace(".mp4", "")
        output_directory = cfg.save_path + f'/{video_name}'
        os.makedirs(output_directory, exist_ok=True)

        torch.save(melspec.squeeze(0).cpu(), os.path.join(output_directory, video_name + '.wav.spec'))


        print('Vocoding')
        audio_gen = vocoder(melspec)
        audio_gen = audio_gen.squeeze()
        audio_gen = audio_gen / 1.1 / audio_gen.abs().max()
        audio_gen = audio_gen.detach().cpu().numpy()
        sf.write(os.path.join(output_directory, video_name + '.wav'), audio_gen, 16000)



        # attach audio to video
        subprocess.call(f"ffmpeg -y -i {video_path} \
                    -i {os.path.join(output_directory, video_name + '.wav')} \
                    -c:v copy -map 0:v:0 -map 1:a:0 \
                    {os.path.join(output_directory, video_name + '.mp4')}", shell=True)

        with open(os.path.join(output_directory, video_name + '.txt'), "w", encoding="utf-8") as f:
            f.write(guidance_text)


if __name__ == '__main__':
    main()



