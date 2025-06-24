<h1 align="center"> CNVSRC2025 VTS Baseline</h1>

## Introduction

This repository is the baseline code for CNVSRC2025 VTS Track.

The code in this repository is based on the SOTA method [Lipvoicer](https://github.com/yochaiye/LipVoicer) on the LRS3 dataset. We have added some configuration files to run this code on CNVSRC.Single. Additionally, we have removed some code that is not needed for running this baseline and modified the implementation of some functionalities.

## Preparation

1. Clone the repository and set up the environment:

```Shell
git clone git@github.com:liu12366262626/CNVSRC2025.git
cd CNVSRC2025/VTS
conda env create -f cnvsrc2025_vts.yaml
# create environment
conda activate cnvsrc2025_vts
```

3. Download and preprocess the dataset. See the instructions in the [preparation](./preparation) folder.

4. Download the [models](#Model-zoo) into path [pretrained_models/](pretrained_models/)

## Logging

We use tensorboard as the logger.

Tensorboard logging files will be writen to `VTS/main_log`

## Training

Please modify the specified `yaml` configuration file in `main.py` to select the training configuration.

### Configuration File

The [conf](conf/) folder lists the configuration files required for this baseline.

Before running any training or testing, please make sure to modify the `code_root_dir` and `data_root_dir` in the corresponding `yaml` file to **the path of this repository** and **the path where the dataset is located**, respectively.

`data.dataset` specifies the path of the dataset and the path of the `.csv` files.

Taking `cncvs` as an example:

1. After data preprocessing is complete, please set `${data_root_dir}` to the parent directory of `cncvs/` and copy [data/cncvs/*.csv](data/cncvs/test.csv) to `${data_root_dir}/cncvs/*.csv`.

2. At this point, the folder structure of `${data_root_dir}` is as follows:

After config all the files, you can 
```Shell
cd CNVSRC2025/VTS/exp/model_v1
# Train video to speech model
bash run.sh

# Train ASR guidance model
cd CNVSRC2025/VTS/exp/model_v2
bash run.sh

# After finishing above steps, you can run inference as following steps:
cd CNVSRC2025/VTS/exp/inference
python infer.py
python compute_vts_metric.py

```



## License

It is noted that the code can only be used for comparative or benchmarking purposes. Users can only use code supplied under a [License](./LICENSE) for non-commercial purposes.

## Contact

```
[Zehua Liu](lzh211[at]bupt.edu.cn)
[CNVSRC2025](cnvsrc[at]cnceleb.org)
```