import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from pystoi.stoi import stoi
import Levenshtein
from funasr import AutoModel
import re
import soundfile as sf
import csv
import json

from tqdm import tqdm
def remove_punctuation(text: str) -> str:
    """
    去除中英文标点，仅保留中文、英文、数字和空白字符
    """
    # 匹配不属于中文、英文、数字和空白的字符
    pattern = r"[^\u4e00-\u9fa5A-Za-z0-9\s]"
    return re.sub(pattern, "", text)


def metric(ref_wav, gen_wav, sample_rate, gt_text, hypothesis):
    min_len = min(len(ref_wav), len(gen_wav))
    ref_wav = ref_wav[:min_len]
    gen_wav = gen_wav[:min_len]

    score = stoi(ref_wav, gen_wav, fs_sig = sample_rate, extended=False)



    # 移除空格，因为CER是基于字符计算的
    reference = gt_text.replace(" ", "")
    hypothesis = hypothesis.replace(" ", "")
    
    # 计算编辑距离
    distance = Levenshtein.distance(reference, hypothesis)
    
    # 计算总的字符数
    total_tokens = len(reference)
    
    # 计算CER
    cer = distance / total_tokens
    
    return {'stoi': score, 'error_token': distance, 'total_token': total_tokens}


if __name__ == "__main__":
    test_csv_file = '/home/liuzehua/task/VTS/LipVoicer_revise/data/CNVSRC_Single/test300.csv'
    gen_wav_prefix = '/home/liuzehua/task/VTS/LipVoicer_revise/main_log/infer_result'
    save_result_path = '/home/liuzehua/task/VTS/LipVoicer_revise/main_log/infer.json'

    # 加载ASR模型，自动下载对应模型和依赖
    model = AutoModel(
        model="paraformer-zh",
        model_revision="v2.0.4",
        vad_model="fsmn-vad",
        vad_model_revision="v2.0.4",
        punc_model="ct-punc-c",
        punc_model_revision="v2.0.4",
        disable_update=True
    )
    
    save_result = []

    with open(test_csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        total_error = 0
        total_token = 0
        avg_stoi = 0
        for row in tqdm(reader):
            audio_path = row[1]
            gt_text = row[4]
            wav_name = os.path.basename(audio_path)
            pred_audio_path = os.path.join(gen_wav_prefix, wav_name.replace('.wav', ''), wav_name)
            
            if not os.path.exists(pred_audio_path):
                print(f'no file {wav_name}')
                continue
            gt_wav, sr_gt = sf.read(audio_path)
            gen_wav, sr_gen = sf.read(pred_audio_path)
            
            gen_text = model.generate(input=pred_audio_path, batch_size_s=300)
            gen_text = remove_punctuation(gen_text[0]['text'])
        
            temp = metric(gt_wav, gen_wav, sr_gt, gt_text, gen_text)
            print(f"{wav_name} , stoi: {temp['stoi']}  cer: {temp['error_token'] / temp['total_token']}")
            
            
            save_result.append(
                {'stoi': temp['stoi'],
                 'cer': temp['error_token']/ temp['total_token'],
                 'pred': gen_text,
                 'gt_text': gt_text}
            )
            total_error = total_error + temp['error_token']
            total_token = total_token + temp['total_token']
            avg_stoi = avg_stoi + temp['stoi']
        
        avg_stoi = avg_stoi / len(save_result)

        save_result.append({
        'total_token': total_token,
        'total_error': total_error,
        'cer': total_error/total_token,
        'stoi': avg_stoi
        }
        )

    # 保存
    with open(save_result_path, 'w', encoding='utf-8') as f:
        json.dump(save_result, f, ensure_ascii=False, indent=4)
                


    

    # wav_files = find_all_wav_files(vts_result_file)
    # for wav_file in wav_files:
    #     wav_name = os.path.basename(wav_file)
    #     gt_wav_path = os.path.join(gt_wav_prefix, wav_name)

    #     gt_wav, sr_gt = sf.read(gt_wav_path)
    #     gen_wav, sr_gen = sf.read(wav_file)
    #     #     # 执行语音转写
    #     # res = model.generate(input='/home/liuzehua/task/VTS/LipVoicer_revise/main_log/infer_result/00026028/00026028.wav', batch_size_s=300)
        
    #     # remove_punctuation(res[0]['text'])


    #     print('show')

