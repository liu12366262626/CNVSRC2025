<h1 align="center"> CNVSRC2025 VTS Baseline</h1>
# CNVSRC2025 - VTS Baseline

This repository provides the baseline implementation for the **Voice from Talking Face (VTS)** track of CNVSRC2025.

## 🧠 Task Description

Given a silent video of a person speaking, the objective is to reconstruct the corresponding speech audio.  
Our baseline follows the diffusion-based method [LipVoicer](https://github.com/yochaiye/LipVoicer), which is currently a strong approach in this field.  
Unlike LipVoicer, which is designed for English, **our implementation is adapted for Chinese**.  
We provide pretrained models and scripts for both training and inference.

---

## 🔧 1. Environment Setup

```bash
git clone git@github.com:liu12366262626/CNVSRC2025.git
cd CNVSRC2025/VTS
conda env create -f cnvsrc2025_vts.yaml
# create environment
conda activate cnvsrc2025_vts
```

## 📁 2. Data Format and pretrained checkpoint preparation

### 2.1 Data Format

Our baseline is trained and evaluated on the **CNVSRC.Single dataset**. The directory structure after preprocessing should look like:
```
cnvsrc-single/
├── dev/
│   ├── audio/              # Audio files (.wav)
│   ├── origin_video/       # Original full-face silent videos (.mp4)
│   └── video/              # Cropped mouth region silent videos (.mp4)
├── eval/
│   ├── origin_video/
│   └── video/
```

We also provide CSV files to organize training/validation/testing splits:
```
VTS/data/CNVSRC_Single/
├── train.csv
├── valid.csv
└── valid300.csv
└── test.csv
```

Each line in the CSV has the following format, please rewrite these csv files according to your download_path:
<video_path>,<audio_path>,<video_frame_count>,<audio_sample_count>,<transcription>
```
#For example:
/home/.../dev/video/00023065.mp4,/home/.../dev/audio/00023065.wav,322,206080,但是老百姓心里或多或少怀疑你到底是否是天命所归的时候为什么又必须要你要做出来这个戏你得装出你是天命所归呢这个就是一个非常有意思的现象啊大家难道不知道你这不是神仙
```

### 2.2 pretrained checkpoint preparation
Please download the checkpoint g_02400000 in:
```
VTS/exp/inference/hifi_gan/
├── config.json
├── env.py
└── g_02400000 (download here)
└── generator.py
└── utils.py
```
Please download the checkpoint checkpoints_ft_lrs3.ckpt(lipvoicer trained on lrs3) and epoch=80.ckpt(CNVSRC2025 VSR Baseline) in:
```
VTS/checkpoint
├── checkpoints_ft_lrs3.ckpt (download here)
├── epoch=80.ckpt (download here)
```


## 🏋️‍♀️ 3. Training & Inference
### 3.1 Stage 1: model_v1 (Classifier-Free Guidance)
This stage follows the vanilla LipVoicer pipeline where a diffusion model generates audio from a silent video and a randomly sampled face image.

**Code:** VTS/exp/model_v1

**Config to modify:** VTS/exp/model_v1/config/train.yaml

**Example configuration:**
```bash
input:
  label_dir: .../VTS/data/CNVSRC_Single # your path
  train_file: train
  valid_file: valid300
...
```

**Run training:**
```bash
cd VTS/exp/model_v1
bash run.sh
```


### 3.2 Stage 2: model_v2 (Classifier Guidance)
In this stage, we train an ASR model that guides the diffusion process to generate clearer and more accurate speech.

**Code:** VTS/exp/model_v2

**Config to modify:** VTS/exp/model_v2/config/train.yaml

**Example configuration:**
```bash
input:
  label_dir: .../VTS/data/CNVSRC_Single # your path
  train_file: train
  valid_file: valid300
  tokenizer_path: .../VTS/exp/model_v2/char_units.txt  # VTS/exp/model_v2/char_units.txt
  pretrained_path: .../VTS/checkpoint/checkpoints_ft_lrs3.ckpt # please download this
...
```

**Run training:**
```bash
cd VTS/exp/model_v2
bash run.sh
```

### 3.3 Inference
After both models are trained, you can run inference using:

**Script**: VTS/exp/inference

**Config**: VTS/exp/inference/config/infer.yaml

**Example configuration:**
```bash
Example configuration:
vtts_path: .../VTS/main_log/temp/vtts_step=67500_val_loss=0.1237.ckpt  # trained model_v1
asr_guidance_path: .../VTS/main_log/temp/asr_step=79920_val_loss=30.8385.ckpt # trained model_v2
vsr_path: .../VTS/checkpoint/epoch=80.ckpt # from CNVSRC2025 VSR Baseline
save_path: .../VTS/main_log/infer_result # you can define this path
split: test # Make sure test.csv is placed under VTS/data/CNVSRC_Single
...
```
**Run Inference:**
```bash
cd VTS/exp/inference
# first generate the speech
python infer.py
# then calculate the VTS metric 
python compute_vts_metric.py \
  --csv /path/to/test.csv \ 
  --gen_wav_dir /path/to/gen_wav \
  --save_result /path/to/save/infer.json 

# csv is your test.csv path
# gen_wav_dir is your generated speech path from infer.py
# save_result is your generated audio path

```

## 📊 4. Baseline Performance
On the CNVSRC.Single evaluation set:

|          Task         |       Training Data           | STOI on Eval| CER on Eval |
|:---------------------:|:-----------------------------:|:-----------:|:-----------:|
|          VTS          |        CNVSRC.Single          |   0.2416    |    31.41%   |


## License

It is noted that the code can only be used for comparative or benchmarking purposes. Users can only use code supplied under a [License](./license) for non-commercial purposes.

## Contact

```
[Zehua Liu](lzh211[at]bupt.edu.cn)
[CNVSRC2025](cnvsrc[at]cnceleb.org)
```


