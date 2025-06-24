
import csv
import os
from scipy.io import wavfile
# CSV 文件路径
csv_file_path = "/home/liuzehua/task/VTS/LipVoicer_revise/data/CNVSRC_Single/test.csv"

save_list = []

# 打开并读取 CSV 文件
with open(csv_file_path, mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)

    # 如果有表头，取消下一行的注释以跳过表头
    # next(reader)

    # 逐行读取并打印
    for row in reader:
        video_path = row[0]
        audio_path = row[0].replace('video', 'audio').replace('.mp4', '.wav')
        temp  = []
        # 检查文件是否存在
        if os.path.exists(audio_path):
            # 读取 wav 文件
            sampling_rate, data = wavfile.read(audio_path)
            
            # 计算采样点数量
            num_samples = data.shape[0]
            save_list.append([video_path, audio_path, row[1], num_samples, row[2]])
        else:
            print(video_path)

# 写入文件
with open('/home/liuzehua/task/VTS/LipVoicer_revise/data/CNVSRC_Single/test1.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # 按行写入
    writer.writerows(save_list)
