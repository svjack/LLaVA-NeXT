import os
import shutil
import argparse
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import torch
import sys
import warnings
from decord import VideoReader, cpu
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")

def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i / fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames, frame_time, video_time

def main(args):
    pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name,
                                                                          torch_dtype="float16",
                                                                          device_map=device_map)
    model.eval()

    max_frames_num = args.max_frames
    force_sample = args.force_sample

    for video_file in tqdm(os.listdir(args.input_path), "run captioning"):
        if video_file.endswith(('.mp4', '.avi', '.mkv')):
            video_path = os.path.join(args.input_path, video_file)
            video, frame_time, video_time = load_video(video_path, max_frames_num, args.fps, force_sample)
            video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().half()
            video = [video]
            conv_template = "qwen_1_5"
            time_instruction = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}. Please answer the following questions related to this video."
            question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruction}\nPlease describe this video in detail."
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
            cont = model.generate(
                input_ids,
                images=video,
                modalities=["video"],
                do_sample=False,
                temperature=0,
                max_new_tokens=4096,
            )
            text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()

            # Save text output
            if not os.path.exists(args.output_path):
                os.makedirs(args.output_path, exist_ok = True)
            output_file_name = os.path.splitext(video_file)[0] + ".txt"
            output_file_path = os.path.join(args.output_path, output_file_name)
            with open(output_file_path, 'w') as f:
                f.write(text_outputs)

            # Copy video to output path
            shutil.copy(video_path, args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert videos to text descriptions.")
    parser.add_argument('--input_path', type=str, required=True, help='Path to input videos')
    parser.add_argument('--output_path', type=str, required=True, help='Path to output text and videos')
    parser.add_argument('--max_frames', type=int, default=19, help='Maximum number of frames to process')
    parser.add_argument('--fps', type=int, default=1, help='Frames per second to sample')
    parser.add_argument('--force_sample', action='store_true', help='Force uniform sampling of frames')
    args = parser.parse_args()
    main(args)
