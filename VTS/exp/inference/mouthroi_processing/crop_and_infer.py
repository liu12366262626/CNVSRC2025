import numpy as np
import os
import cv2
import time
import shutil
import subprocess
import tempfile
import torch
import torchvision
from .pipelines.pipeline import InferencePipeline


def write_video_ffmpeg(rois, target_path, ffmpeg):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    decimals = 10
    fps = 25
    tmp_dir = tempfile.mkdtemp()
    for i_roi, roi in enumerate(rois):
        cv2.imwrite(os.path.join(tmp_dir, str(i_roi).zfill(decimals)+'.png'), roi)
    list_fn = os.path.join(tmp_dir, "list")
    with open(list_fn, 'w') as fo:
        fo.write("file " + "'" + tmp_dir+'/%0'+str(decimals)+'d.png' + "'\n")
    ## ffmpeg
    if os.path.isfile(target_path):
        os.remove(target_path)
    cmd = [ffmpeg, "-f", "concat", "-safe", "0", "-i", list_fn, "-q:v", "1", "-r", str(fps), '-y', '-crf', '20', target_path]
    pipe = subprocess.run(cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    # rm tmp dir
    shutil.rmtree(tmp_dir)
    return


def main(video_filename, output_directory, cfg):

    # Verify that the videos sample rate is 25Hz
    fps = cv2.VideoCapture(video_filename).get(cv2.CAP_PROP_FPS)
    if fps != 25:
        video_filename25 = video_filename.replace(".mp4", "25fps.mp4")
        print("Converting fps to 25Hz")
        os.system(f"ffmpeg -y -i {video_filename} -filter:v fps=fps=25 {video_filename25}")
        os.system(f"mv {video_filename25} {video_filename}")

    # # 记录开始时间
    # start_time = time.time()
    # print("Cropping mouth region")
    # detector = "retinaface"
    # config_filename = "mouthroi_processing/configs/LRS3_V_WER19.1.ini"
    pipeline = InferencePipeline(config_filename, device='cuda', detector=detector, face_track=True, cfg=cfg)
    pipeline.to('cuda')
    # landmarks = pipeline.process_landmarks(video_filename, landmarks_filename=None)
    # video = pipeline.dataloader.load_video(video_filename)
    # mouth_crop = pipeline.dataloader.video_process(video, landmarks)

    # # write_video_ffmpeg(mouth_crop, dst_filename, "/usr/bin/ffmpeg")
    # dst_filename = video_filename.split('/')[-1].replace(".mp4", "_mouthcrop.mp4")
    # dst_path = os.path.join(output_directory, dst_filename)
    # write_video_ffmpeg(mouth_crop, dst_path, "ffmpeg")
    # mouth_crop = torch.tensor(mouth_crop)
    # mouth_crop = mouth_crop.unsqueeze(1)
    
    # video = torchvision.io.read_video(dst_path, pts_unit='sec')[0] # T H W C
    # video = video.permute(0, 3, 1, 2).contiguous().to('cuda') # T C H W
    # _mouth_crop = pipeline.dataloader.video_transform(mouth_crop).to('cuda')
    
    # # 记录crop mouth region的结束时间
    # crop_end_time = time.time()
    # crop_duration = crop_end_time - start_time
    # print(f"Crop mouth region took {crop_duration:.2f} seconds")

    # print("Performing lip-reading")
    transcript = pipeline.model.infer(_mouth_crop)
    
    # # 记录lipreading的结束时间
    # lipreading_end_time = time.time()
    # lipreading_duration = lipreading_end_time - crop_end_time
    # print(f"Lipreading took {lipreading_duration:.2f} seconds")
    
    # with open(os.path.join(output_directory, "lipreading_prediction.txt"), 'w', encoding='utf-8') as f:
    #     f.write(transcript)
    return transcript


if __name__ == "__main__":
    data_filename = "/workspace/inputs/yochai/lipvoicer/data/raw/LRS3/test/0Fi83BHQsMA/00002.mp4"
    dst_filename = "./out.mp4"
    main(data_filename, dst_filename)
