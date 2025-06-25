import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import argparse
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
    pattern = r"[^\u4e00-\u9fa5A-Za-z0-9\s]"
    return re.sub(pattern, "", text)

def metric(ref_wav, gen_wav, sample_rate, gt_text, hypothesis):
    min_len = min(len(ref_wav), len(gen_wav))
    ref_wav = ref_wav[:min_len]
    gen_wav = gen_wav[:min_len]

    score = stoi(ref_wav, gen_wav, fs_sig=sample_rate, extended=False)

    reference = gt_text.replace(" ", "")
    hypothesis = hypothesis.replace(" ", "")

    distance = Levenshtein.distance(reference, hypothesis)
    total_tokens = len(reference)
    cer = distance / total_tokens

    return {'stoi': score, 'error_token': distance, 'total_token': total_tokens}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VTS output using STOI and CER")
    parser.add_argument('--csv', type=str, required=True, help='Path to the test CSV file')
    parser.add_argument('--gen_wav_dir', type=str, required=True, help='Path to generated wav directory')
    parser.add_argument('--save_result', type=str, required=True, help='Path to save result JSON file')
    args = parser.parse_args()

    test_csv_file = args.csv
    gen_wav_prefix = args.gen_wav_dir
    save_result_path = args.save_result

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

            save_result.append({
                'stoi': temp['stoi'],
                'cer': temp['error_token'] / temp['total_token'],
                'pred': gen_text,
                'gt_text': gt_text
            })

            total_error += temp['error_token']
            total_token += temp['total_token']
            avg_stoi += temp['stoi']

        avg_stoi /= len(save_result)

        save_result.append({
            'total_token': total_token,
            'total_error': total_error,
            'cer': total_error / total_token,
            'stoi': avg_stoi
        })

    with open(save_result_path, 'w', encoding='utf-8') as f:
        json.dump(save_result, f, ensure_ascii=False, indent=4)
