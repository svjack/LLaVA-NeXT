<p align="center" width="100%">
<img src="https://i.postimg.cc/pL17YtG4/WX20240508-220230-2x.png"  width="80%" height="80%">
</p>

# LLaVA-NeXT: Open Large Multimodal Models
[![Static Badge](https://img.shields.io/badge/llava_video-paper-green)](http://arxiv.org/abs/2410.02713)
[![Static Badge](https://img.shields.io/badge/llava_onevision-paper-green)](https://arxiv.org/abs/2408.03326)
[![llava_next-blog](https://img.shields.io/badge/llava_next-blog-green)](https://llava-vl.github.io/blog/)

[![llava_onevision-demo](https://img.shields.io/badge/llava_onevision-demo-red)](https://llava-onevision.lmms-lab.com/)
[![llava_next-video_demo](https://img.shields.io/badge/llava_video-demo-red)](https://huggingface.co/spaces/WildVision/vision-arena)
[![llava_next-interleave_demo](https://img.shields.io/badge/llava_next-interleave_demo-red)](https://huggingface.co/spaces/lmms-lab/LLaVA-NeXT-Interleave-Demo)
[![Openbayes Demo](https://img.shields.io/static/v1?label=Demo&message=OpenBayes%E8%B4%9D%E5%BC%8F%E8%AE%A1%E7%AE%97&color=green)](https://openbayes.com/console/public/tutorials/gW0ng9jKXfO)

[![llava_video-checkpoints](https://img.shields.io/badge/llava_video-checkpoints-blue)](https://huggingface.co/collections/lmms-lab/llava-next-video-661e86f5e8dabc3ff793c944)
[![llava_onevision-checkpoints](https://img.shields.io/badge/llava_onevision-checkpoints-blue)](https://huggingface.co/collections/lmms-lab/llava-onevision-66a259c3526e15166d6bba37)
[![llava_next-interleave_checkpoints](https://img.shields.io/badge/llava_next-interleave_checkpoints-blue)](https://huggingface.co/collections/lmms-lab/llava-next-interleave-66763c55c411b340b35873d1)
[![llava_next-image_checkpoints](https://img.shields.io/badge/llava_next-image_checkpoints-blue)](https://huggingface.co/lmms-lab)

# LLaVA-Video-7B-Qwen2 and LLaVA-NeXT

This repository provides instructions and scripts to set up and run the LLaVA-Video-7B-Qwen2 model using the LLaVA-NeXT framework. The process involves cloning the repository, setting up the environment, and running a video captioning script. Additionally, it includes steps to generate metadata and upload the dataset to Hugging Face.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Video Captioning Script](#running-the-video-captioning-script)
- [Generating Metadata](#generating-metadata)
- [Uploading to Hugging Face](#uploading-to-hugging-face)

## Prerequisites

Before you begin, ensure you have the following installed:
- `git-lfs`
- `ffmpeg`
- `cbm`

You can install these dependencies using the following command:

```bash
sudo apt-get update && sudo apt-get install git-lfs ffmpeg cbm
sudo apt-get upgrade ffmpeg
```

## Installation

1. Clone the LLaVA-NeXT repository:

```bash
git clone https://github.com/svjack/LLaVA-NeXT
cd LLaVA-NeXT
```

2. Set up the conda environment:

```bash
conda create -n llava python=3.10 -y
conda activate llava
```

3. Install the necessary Python packages:

```bash
pip install ipykernel
python -m ipykernel install --user --name llava --display-name "llava"
pip install --upgrade pip  # Enable PEP 660 support.
pip install torch
pip install -e ".[train]"
pip install flash_attn==2.3.4 --no-build-isolation
pip install moviepy=="1.0.3"
```

## Running the Video Captioning Script

### To run the video captioning script, use the following command: 
### videos in dir have length 6s and [640, 360] as resolution 

- length 6s
### (refer to https://github.com/svjack/WatermarkRemover: python .\video_skipper.py  --input  .\原神风景视频（去水印）1920x1080_人物 -s 5 -e 10 -m 6)
- Resize [640, 360]
```python
import os
from moviepy.editor import VideoFileClip

def resize_videos_in_folder(input_folder, output_folder, new_size=(640, 360)):
    """
    将输入文件夹中的所有视频文件调整大小并保存到输出文件夹中。

    :param input_folder: 输入文件夹路径，包含要调整大小的视频文件。
    :param output_folder: 输出文件夹路径，用于保存调整大小后的视频文件。
    :param new_size: 新的视频尺寸，格式为 (width, height)。
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 检查文件是否为视频文件（这里假设视频文件为常见的视频格式）
        if filename.endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 使用 moviepy 加载视频文件
            video_clip = VideoFileClip(input_path)

            # 调整视频大小
            resized_clip = video_clip.resize(new_size)

            # 保存调整大小后的视频
            resized_clip.write_videofile(output_path, codec='libx264')

            # 关闭视频剪辑以释放资源
            video_clip.close()
            resized_clip.close()

# 示例用法
input_folder = 'C:/Users/DELL/Downloads/WatermarkRemover-master/WatermarkRemover-master/原神风景视频（去水印）1920x1080_人物_skip/'
output_folder = 'C:/Users/DELL/Downloads/WatermarkRemover-master/WatermarkRemover-master/原神风景视频（去水印）1920x1080_人物_processed'
new_size = (640, 360)  # 新的视频尺寸

resize_videos_in_folder(input_folder, output_folder, new_size)
```


```bash
python llava_qwen_video_caption.py --input_path "原神风景视频（去水印）1920x1080_人物_resized" --output_path "原神风景视频（去水印）1920x1080_人物_captioned" --max_frames 19 --fps 1 --force_sample
```

## Generating Metadata

After running the captioning script, you can generate metadata for the processed videos. The following script will create a `metadata.csv` file:

```bash
#!/bin/bash

# 源目录和目标目录
src_dir="Toradora_Videos_Omni_Captioned"
dst_dir0="Toradora_Videos_Omni_Captioned_0"
dst_dir1="Toradora_Videos_Omni_Captioned_1"

# 创建目标目录
mkdir -p "$dst_dir0" "$dst_dir1"

# 使用带引号的变量和IFS处理文件名中的空格
IFS=$'\n'

# 获取所有.mp4文件并按文件名排序
files=($(find "$src_dir" -maxdepth 1 -name "*.mp4" -print0 | sort -z | xargs -0 printf "%s\n"))

# 计算文件总数和中间点
total_files=${#files[@]}
half_point=$((total_files / 2))

# 复制前一半到dst_dir0
for ((i=0; i<half_point; i++)); do
    file="${files[i]}"
    base_name=$(basename "$file" .mp4)
    cp "$file" "$dst_dir0/"
    txt_file="$src_dir/$base_name.txt"
    if [ -f "$txt_file" ]; then
        cp "$txt_file" "$dst_dir0/"
    fi
done

# 复制后一半到dst_dir1
for ((i=half_point; i<total_files; i++)); do
    file="${files[i]}"
    base_name=$(basename "$file" .mp4)
    cp "$file" "$dst_dir1/"
    txt_file="$src_dir/$base_name.txt"
    if [ -f "$txt_file" ]; then
        cp "$txt_file" "$dst_dir1/"
    fi
done

echo "文件已成功分割并复制到目标目录"
```

```python
import pathlib
import pandas as pd

def r_func(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def generate_metadata(input_dir):
    # 创建Path对象并标准化路径
    input_path = pathlib.Path(input_dir).resolve()
    
    # 收集所有视频和文本文件
    file_list = []
    for file_path in input_path.rglob("*"):
        if file_path.suffix.lower() in ('.mp4', '.txt'):
            file_list.append({
                "stem": file_path.stem,
                "path": file_path,
                "type": "video" if file_path.suffix.lower() == '.mp4' else "text"
            })
    
    # 创建DataFrame并分组处理
    df = pd.DataFrame(file_list)
    grouped = df.groupby('stem')
    
    metadata = []
    for stem, group in grouped:
        # 获取组内文件
        videos = group[group['type'] == 'video']
        texts = group[group['type'] == 'text']
        
        # 确保每组有且只有一个视频和一个文本文件
        if len(videos) == 1 and len(texts) == 1:
            video_path = videos.iloc[0]['path']
            text_path = texts.iloc[0]['path']
            
            metadata.append({
                "file_name": video_path.name,  # 自动处理不同系统的文件名
                "prompt": r_func(text_path)
            })
    
    # 保存结果到CSV
    output_path = input_path.parent / "metadata.csv"
    pd.DataFrame(metadata).to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Metadata generated at: {output_path}")

'''
---
configs:
- config_name: default
  data_files:
  - split: train
    path: 
    - "*.mp4"
    - "metadata.csv"
---
'''
```

```python
pip install -U datasets

def r_func(x):
    with open(x, "r") as f:
        return f.read().strip()

import pathlib
import pandas as pd
import numpy as np

pd.DataFrame(
    pd.DataFrame(
        pd.Series(
            list(pathlib.Path("C:/Users/DELL/Downloads/原神风景视频（去水印）1920x1080_人物_captioned/").rglob("*"))
        ).map(
            str
        ).map(lambda x: x if x.endswith("mp4") or x.endswith("txt") else np.nan).dropna().map(
            lambda x: (x.split("\\")[-1].split(".")[0], x)
        ).values.tolist()
    ).groupby(0)[1].apply(list).map(
        lambda l: list(map(lambda x: x.split("\\")[-1] if x.endswith(".mp4") else x, l))
    ).map(
        lambda l: {
            "file_name": list(filter(lambda x: x.endswith(".mp4"), l))[0],
            "prompt": r_func(list(filter(lambda x: x.endswith(".txt"), l))[0]),
        }
    ).values.tolist()
).to_csv("metadata.csv", index=False)

!cp metadata.csv 原神风景视频（去水印）1920x1080_人物_captioned
```

- OR Use function like

```python
import os
import shutil
import uuid
import pandas as pd
from pathlib import Path

def r_func(x):
    with open(x, "r", encoding="utf-8") as f:
        return f.read().strip()

def process_files(input_path, output_path, prefix=""):
    # 创建输出路径
    os.makedirs(output_path, exist_ok=True)

    # 获取所有mp4和txt文件
    files = list(Path(input_path).rglob("*"))
    mp4_files = [str(f) for f in files if f.suffix == ".mp4"]
    txt_files = [str(f) for f in files if f.suffix == ".txt"]

    # 创建一个字典来存储成对儿的文件
    file_pairs = {}

    # 遍历mp4文件，找到对应的txt文件
    for mp4_file in mp4_files:
        base_name = Path(mp4_file).stem
        txt_file = next((f for f in txt_files if Path(f).stem == base_name), None)
        if txt_file:
            file_pairs[base_name] = (mp4_file, txt_file)

    # 创建metadata列表
    metadata = []

    # 处理每一对文件
    for base_name, (mp4_file, txt_file) in file_pairs.items():
        # 生成UUID
        unique_id = str(uuid.uuid4())

        # 构建新的文件名
        new_mp4_file = Path(output_path) / f"{unique_id}.mp4"
        new_txt_file = Path(output_path) / f"{unique_id}.txt"

        # 拷贝mp4文件到新路径并重命名
        shutil.copy(mp4_file, new_mp4_file)

        # 读取txt文件内容
        prompt = r_func(txt_file)

        # 在内容前添加prefix
        modified_prompt = f"{prefix}{prompt}"

        # 将修改后的内容写入新的txt文件
        with open(new_txt_file, "w", encoding="utf-8") as f:
            f.write(modified_prompt)

        # 添加到metadata列表
        metadata.append({
            "file_name": f"{unique_id}.mp4",
            "prompt": modified_prompt,
            "original_file_name": base_name,  # 添加重命名前的文件名
        })

    # 生成metadata.csv文件
    df = pd.DataFrame(metadata)
    df.to_csv(Path(output_path) / "metadata.csv", index=False)

# 示例调用
input_path = "风物集_captioned"
output_path = "风物集_processed"
process_files(input_path, output_path)
```
- OR
```python
import os
import shutil
import uuid
import pandas as pd
from pathlib import Path
from tqdm import tqdm  # 导入 tqdm 用于显示进度条

def r_func(x):
    with open(x, "r", encoding="utf-8") as f:
        return f.read().strip()

def process_files(input_path, output_path, evaluation_results_path, prefix=""):
    # 创建输出路径
    os.makedirs(output_path, exist_ok=True)

    # 读取 evaluation_results.csv 文件
    evaluation_df = pd.read_csv(evaluation_results_path)

    # 创建metadata列表
    metadata = []

    # 使用 tqdm 包装循环，显示进度条
    for index, row in tqdm(evaluation_df.iterrows(), total=len(evaluation_df), desc="Processing files"):
        video_name = row['video_name']
        mp4_file = Path(input_path) / f"{video_name}"
        txt_file = Path(input_path) / f"{Path(video_name).stem}.txt"

        # 检查视频文件和文本文件是否存在
        if mp4_file.exists() and txt_file.exists():
            # 生成UUID
            unique_id = str(uuid.uuid4())

            # 构建新的文件名
            new_mp4_file = Path(output_path) / f"{unique_id}.mp4"
            new_txt_file = Path(output_path) / f"{unique_id}.txt"

            # 拷贝mp4文件到新路径并重命名
            shutil.copy(mp4_file, new_mp4_file)

            # 读取txt文件内容
            prompt = r_func(txt_file)

            # 在内容前添加prefix
            modified_prompt = f"{prefix}{prompt}"

            # 将修改后的内容写入新的txt文件
            with open(new_txt_file, "w", encoding="utf-8") as f:
                f.write(modified_prompt)

            # 添加到metadata列表
            metadata_entry = {
                "file_name": f"{unique_id}.mp4",
                "prompt": modified_prompt,
                "original_file_name": video_name,  # 添加重命名前的文件名
            }

            # 添加 evaluation_results.csv 中对应行的所有键值（除了 video_name）
            for key, value in row.items():
                if key != 'video_name':
                    metadata_entry[key] = value

            metadata.append(metadata_entry)

    # 生成metadata.csv文件
    df = pd.DataFrame(metadata)
    df.to_csv(Path(output_path) / "metadata.csv", index=False)

# 示例调用
input_path = "../videos_captioned/"
output_path = "../videos_upload/"
evaluation_results_path = "evaluation_results.csv"  # 可输入参数
process_files(input_path, output_path, evaluation_results_path)

from moviepy.editor import VideoFileClip
import numpy as np

def detect_black_scenes(video_path, black_threshold=0.1, frame_threshold=0.9):
    """
    检测视频中是否存在黑色场景。

    :param video_path: 视频文件的路径
    :param black_threshold: 判断为黑色的亮度阈值（0-1之间），默认0.1
    :param frame_threshold: 判断为黑色场景的帧比例阈值（0-1之间），默认0.9
    :return: 如果视频中存在黑色场景，返回 True；否则返回 False
    """
    # 加载视频
    clip = VideoFileClip(video_path)
    
    # 初始化黑色场景计数器
    black_frames = 0
    total_frames = 0
    
    # 遍历视频的每一帧
    for frame in clip.iter_frames():
        total_frames += 1
        # 计算帧的平均亮度
        brightness = np.mean(frame) / 255  # 将亮度归一化到 0-1 范围
        if brightness < black_threshold:
            black_frames += 1
    
    # 计算黑色帧的比例
    black_frame_ratio = black_frames / total_frames
    return black_frame_ratio
    # 判断是否存在黑色场景
    if black_frame_ratio >= frame_threshold:
        return True
    else:
        return False

df = pd.read_csv("../videos_upload/metadata.csv")
from tqdm import tqdm 
import os
req = []
for i, r in tqdm(df.iterrows()):
    d = r.to_dict()
    d["black_ratio"] = detect_black_scenes(os.path.join("../videos_upload", d["file_name"]))
    req.append(d)

df_with_score = pd.DataFrame(req)
df_with_score.to_csv("../metadata.csv", index = False)

!cp ../metadata.csv videos_upload
!huggingface_cli login
!huggingface-cli upload svjack/Genshin-Impact-Cutscenes-with-score-organized videos_upload . --repo-type dataset
```

## Uploading to Hugging Face

1. Load the dataset using the `datasets` library:

```python
from datasets import load_dataset
video_ds = load_dataset("videofolder", data_dir="原神风景视频（去水印）1920x1080_人物_captioned/")
video_ds
```

2. Log in to Hugging Face CLI:

```bash
!huggingface-cli login
```

3. Upload the dataset to Hugging Face:

```bash
svjack/video-dataset-genshin-impact-landscape-organized
```

4. Upload the `metadata.csv` file and create a `readme.md` file similar to the example provided in [sayakpaul/video-dataset-disney-organized](https://huggingface.co/datasets/sayakpaul/video-dataset-disney-organized).

## Conclusion

This guide provides a step-by-step process to set up and run the LLaVA-Video-7B-Qwen2 model using the LLaVA-NeXT framework. It also includes instructions on how to generate metadata and upload the dataset to Hugging Face.

## Release Notes

- **[2024/10/04] 🔥 LLaVA-Video** (formerly LLaVA-NeXT-Video) has undergone a major upgrade! We are excited to release **LLaVA-Video-178K**, a high-quality synthetic dataset for video instruction tuning. This dataset includes:

  - 178,510 caption entries
  - 960,792 open-ended Q&A pairs
  - 196,198 multiple-choice Q&A items

  Along with this, we’re also releasing the **LLaVA-Video 7B/72B models**, which deliver competitive performance on the latest video benchmarks, including [Video-MME](https://video-mme.github.io/home_page.html#leaderboard), [LongVideoBench](https://longvideobench.github.io/), and [Dream-1K](https://tarsier-vlm.github.io/).

  📄 **Explore more**:
  - [LLaVA-Video-178K Dataset](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K): Download the dataset.
  - [LLaVA-Video Models](https://huggingface.co/collections/lmms-lab/llava-video-661e86f5e8dabc3ff793c944): Access model checkpoints.
  - [Paper](http://arxiv.org/abs/2410.02713): Detailed information about LLaVA-Video.
  - [LLaVA-Video Documentation](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/docs/LLaVA_Video_1003.md): Guidance on training, inference and evaluation.

- [2024/09/13] 🔥 **🚀 [LLaVA-OneVision-Chat](docs/LLaVA_OneVision_Chat.md)**. The new LLaVA-OV-Chat (7B/72B) significantly improves the chat experience of LLaVA-OV. 📄
  
  ![](docs/ov_chat_images/chat_results.png)

- [2024/08/06] 🔥 **🚀 [LLaVA-OneVision (OV)](https://llava-vl.github.io/blog/2024-08-05-llava-onevision/)!** The new LLaVA-OV models (0.5B/7B/72B) achieve new state-of-the-art performance across single-image, multi-image, and video benchmarks, sometimes rivaling top commercial models on 47 diverse benchmarks. 📄 Explore More:
  * [[Paper]](https://arxiv.org/abs/2408.03326): In-depth insights, new emegerging scenarios, ie, strong video understadning through task transfer from images.
  * [[LLaVA-OV Doc]](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/docs/LLaVA_OneVision.md): Model inference and evaluation guidance.
  * [[Scripts]](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/scripts/train): Start training models on your single-image/multi-image/video data.
    
- [2024/07/16] 🔥 **LLaVA-NeXT-Video** has been upgraded. The new 32B model achieves the best open-source performance on several video benchmarks, including [Video-MME](https://video-mme.github.io/home_page.html#leaderboard). Please refer to [this page](docs/LLaVA-NeXT-Video_0716.md) for details, refer to [llava_next-video_demo](https://huggingface.co/spaces/WildVision/vision-arena) for demo.


- [2024/06/23] 🔥 **LLaVA-NeXT-Interleave** is released. We utilize image-text interleaved format to unify multi-image, video, and 3D tasks in one LLM and achieve **SoTA** performance on a wide range of benchmarks. Check out [paper](https://arxiv.org/pdf/2407.07895), [blog](https://llava-vl.github.io/blog/2024-06-16-llava-next-interleave/), and [checkpoints](https://huggingface.co/collections/lmms-lab/llava-next-interleave-66763c55c411b340b35873d1) to see new capabilities and improved performance! We have released 0.5b, 7b, and 7b-dpo models.
  * An all-round LLM for multi-image, video, and 3D with strong performance \[[demo](https://huggingface.co/spaces/lmms-lab/LLaVA-NeXT-Interleave-Demo)\]
  * Construct interleave training data [**M4-Instruct**](https://huggingface.co/datasets/lmms-lab/M4-Instruct-Data)
  * Construct multi-image benchmark [**LLaVA-Interleave Bench**](https://huggingface.co/datasets/lmms-lab/LLaVA-NeXT-Interleave-Bench)


- [2024/05/25] 🔥 Wondering "[What Else Influences Visual Instruction Tuning Beyond Data?](https://llava-vl.github.io/blog/2024-05-25-llava-next-ablations/)" Our new [blog](https://llava-vl.github.io/blog/2024-05-25-llava-next-ablations/) summarizes empirical explorations to ablate the various design choices in improving LMMs except instruct data itself. Meanwhile, open-source the recapioned high-quality data using LLaVA-NeXT-34B on [[COCO]](https://huggingface.co/datasets/lmms-lab/LLaVA-ReCap-118K) [[LCS]](https://huggingface.co/datasets/lmms-lab/LLaVA-ReCap-558K) [[CC3M]](https://huggingface.co/datasets/lmms-lab/LLaVA-ReCap-CC3M).
  * Architectures (LMM & Vision Encoder)
  * Visual Representations (Resolution & # Tokens)
  * Training Strategies (High-quality data & Trainable modules)

- [2024/05/10] 🔥 **LLaVA-NeXT** (Stronger) models are released, with support of stronger LMM inlcuding LLama-3 (8B) and Qwen-1.5 (72B/110B) Check out [[blog](https://llava-vl.github.io/blog/2024-05-10-llava-next-stronger-llms/)] and [[checkpoints](https://huggingface.co/lmms-lab)] to see improved performance!
- [2024/05/10] 🔥 **LLaVA-NeXT** (Video) is released. The image-only-trained LLaVA-NeXT model is surprisingly strong on video tasks with zero-shot modality transfer. DPO training with AI feedback on videos can yield significant improvement. [[Blog](https://llava-vl.github.io/blog/2024-04-30-llava-next-video/)], [[checkpoints](https://huggingface.co/collections/lmms-lab/llava-next-video-661e86f5e8dabc3ff793c944)] and [[sglang](https://github.com/sgl-project/sglang)]
- [2024/01/30] 🔥 **LLaVA-NeXT** is out! With additional scaling to LLaVA-1.5, LLaVA-NeXT-34B outperforms Gemini Pro on some benchmarks. It can now process 4x more pixels and perform more tasks/applications than before. Check out the [blog post](https://llava-vl.github.io/blog/2024-01-30-llava-next/), and explore the [demo](https://llava.hliu.cc/)! Models are available in [Model Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md). Training/eval data and scripts coming soon.
<details>
<summary>More</summary>
  
- [2024/03/10] 🔥 Releasing **LMMs-Eval**, a highly efficient evaluation pipeline we used when developing LLaVA-NeXT. It supports the evaluation of LMMs on dozens of public datasets and allows new dataset onboarding, making the dev of new LMMs much faster. [[Blog](https://lmms-lab.github.io/lmms-eval-blog/lmms-eval-0.1/)] [[Codebase](https://github.com/EvolvingLMMs-Lab/lmms-eval)]
  
- [2023/11/10] [LLaVA-Plus](https://llava-vl.github.io/llava-plus/) is released: Learning to Use Tools for Creating Multimodal Agents, with LLaVA-Plus (LLaVA that Plug and Learn to Use Skills). [[Project Page](https://llava-vl.github.io/llava-plus/)] [[Demo](https://llavaplus.ngrok.io/)] [[Code](https://github.com/LLaVA-VL/LLaVA-Plus-Codebase)] [[Paper](https://arxiv.org/abs/2311.05437)]
- [2023/11/02] [LLaVA-Interactive](https://llava-vl.github.io/llava-interactive/) is released: Experience the future of human-AI multimodal interaction with an all-in-one demo for Image Chat, Segmentation, Generation and Editing. [[Project Page](https://llava-vl.github.io/llava-interactive/)] [[Demo](https://llavainteractive.ngrok.io/)] [[Code](https://github.com/LLaVA-VL/LLaVA-Interactive-Demo)] [[Paper](https://arxiv.org/abs/2311.00571)]
- [2023/10/26] 🔥 LLaVA-1.5 with LoRA achieves comparable performance as full-model finetuning, with a reduced GPU RAM requirement ([ckpts](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md#llava-v15), [script](https://github.com/haotian-liu/LLaVA#train)). We also provide a [doc](https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md) on how to finetune LLaVA-1.5 on your own dataset with LoRA.
- [2023/10/12] Check out the Korean LLaVA (Ko-LLaVA), created by ETRI, who has generously supported our research! [[🤗 Demo](https://huggingface.co/spaces/etri-vilab/Ko-LLaVA)]
- [2023/10/05] 🔥 LLaVA-1.5 is out! Achieving SoTA on 11 benchmarks, with just simple modifications to the original LLaVA, utilizes all public data, completes training in ~1 day on a single 8-A100 node, and surpasses methods like Qwen-VL-Chat that use billion-scale data. Check out the [technical report](https://arxiv.org/abs/2310.03744), and explore the [demo](https://llava.hliu.cc/)! Models are available in [Model Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md). The training data and scripts of LLaVA-1.5 are released [here](https://github.com/haotian-liu/LLaVA#train), and evaluation scripts are released [here](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md)!
- [2023/09/26] LLaVA is improved with reinforcement learning from human feedback (RLHF) to improve fact grounding and reduce hallucination. Check out the new SFT and RLHF checkpoints at project [[LLavA-RLHF]](https://llava-rlhf.github.io/)
- [2023/09/22] [LLaVA](https://arxiv.org/abs/2304.08485) is accepted by NeurIPS 2023 as **oral presentation**, and [LLaVA-Med](https://arxiv.org/abs/2306.00890) is accepted by NeurIPS 2023 Datasets and Benchmarks Track as **spotlight presentation**.
- [2023/11/06] Support **Intel** dGPU and CPU platforms. [More details here.](https://github.com/haotian-liu/LLaVA/tree/intel/docs/intel)
- [2023/10/12] LLaVA is now supported in [llama.cpp](https://github.com/ggerganov/llama.cpp/pull/3436) with 4-bit / 5-bit quantization support!
- [2023/10/11] The training data and scripts of LLaVA-1.5 are released [here](https://github.com/haotian-liu/LLaVA#train), and evaluation scripts are released [here](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md)!
- [2023/10/10] [Roboflow Deep Dive](https://blog.roboflow.com/first-impressions-with-llava-1-5/): First Impressions with LLaVA-1.5.
- [2023/09/20] We summarize our empirical study of training 33B and 65B LLaVA models in a [note](https://arxiv.org/abs/2309.09958). Further, if you are interested in the comprehensive review, evolution and trend of multimodal foundation models, please check out our recent survey paper [``Multimodal Foundation Models: From Specialists to General-Purpose Assistants''.](https://arxiv.org/abs/2309.10020)
<p align="center">
  <img src="https://github.com/Computer-Vision-in-the-Wild/CVinW_Readings/blob/main/images/mfm_evolution.jpeg?raw=true" width=50%/>
</p>

- [2023/07/19] 🔥 We release a major upgrade, including support for LLaMA-2, LoRA training, 4-/8-bit inference, higher resolution (336x336), and a lot more. We release [LLaVA Bench](https://github.com/haotian-liu/LLaVA/blob/main/docs/LLaVA_Bench.md) for benchmarking open-ended visual chat with results from Bard and Bing-Chat. We also support and verify training with RTX 3090 and RTX A6000. Check out [LLaVA-from-LLaMA-2](https://github.com/haotian-liu/LLaVA/blob/main/docs/LLaVA_from_LLaMA2.md), and our [model zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md)!
- [2023/06/26] [CVPR 2023 Tutorial](https://vlp-tutorial.github.io/) on **Large Multimodal Models: Towards Building and Surpassing Multimodal GPT-4**!  Please check out [[Slides](https://datarelease.blob.core.windows.net/tutorial/vision_foundation_models_2023/slides/Chunyuan_cvpr2023_tutorial_lmm.pdf)] [[Notes](https://arxiv.org/abs/2306.14895)] [[YouTube](https://youtu.be/mkI7EPD1vp8)] [[Bilibli](https://www.bilibili.com/video/BV1Ng4y1T7v3/)].
- [2023/06/11] We released the preview for the most requested feature: DeepSpeed and LoRA support!  Please see documentations [here](./docs/LoRA.md).
- [2023/06/01] We released **LLaVA-Med: Large Language and Vision Assistant for Biomedicine**, a step towards building biomedical domain large language and vision models with GPT-4 level capabilities.  Checkout the [paper](https://arxiv.org/abs/2306.00890) and [page](https://github.com/microsoft/LLaVA-Med).
- [2023/05/06] We are releasing [LLaVA-Lighting-MPT-7B-preview](https://huggingface.co/liuhaotian/LLaVA-Lightning-MPT-7B-preview), based on MPT-7B-Chat!  See [here](#LLaVA-MPT-7b) for more details.
- [2023/05/02] 🔥 We are releasing LLaVA-Lighting!  Train a lite, multimodal GPT-4 with just $40 in 3 hours!  See [here](#train-llava-lightning) for more details.
- [2023/04/27] Thanks to the community effort, LLaVA-13B with 4-bit quantization allows you to run on a GPU with as few as 12GB VRAM!  Try it out [here](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/llava).
- [2023/04/17] 🔥 We released **LLaVA: Large Language and Vision Assistant**. We propose visual instruction tuning, towards building large language and vision models with GPT-4 level capabilities.  Checkout the [paper](https://arxiv.org/abs/2304.08485) and [demo](https://llava.hliu.cc/).

</details>

<!-- <a href="https://llava.hliu.cc/"><img src="assets/demo.gif" width="70%"></a> -->

**Usage and License Notices**: This project utilizes certain datasets and checkpoints that are subject to their respective original licenses. Users must comply with all terms and conditions of these original licenses, including but not limited to the [OpenAI Terms of Use](https://openai.com/policies/terms-of-use) for the dataset and the specific licenses for base language models for checkpoints trained using the dataset (e.g. [Llama-1/2 community license](https://ai.meta.com/llama/license/) for LLaMA-2 and Vicuna-v1.5, [Tongyi Qianwen RESEARCH LICENSE AGREEMENT](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat/blob/main/LICENSE) and [Llama-3 Research License](https://llama.meta.com/llama3/license/)). This project does not impose any additional constraints beyond those stipulated in the original licenses. Furthermore, users are reminded to ensure that their use of the dataset and checkpoints is in compliance with all applicable laws and regulations.

## Models & Scripts

### Installation

#### 1. **Clone this repository and navigate to the LLaVA folder:**
```bash
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
cd LLaVA-NeXT
```

#### 2. **Install the inference package:**
```bash
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"
```

### Project Navigation
Please checkout the following page for more inference & evaluation details.

#### - **LLaVA-OneVision: Easy Task Transfer**
- [LLaVA-OneVision]([./docs/LLaVA-NeXT.md](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/docs/LLaVA_OneVision.md)): for demo inference. The evaluation code is in [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).

#### - **LLaVA-NeXT: Stronger LLMs Supercharge Multimodal Capabilities in the Wild**
- [LLaVA-NeXT-Image](./docs/LLaVA-NeXT.md): for image demo inference and evaluation of stronger LMMs using [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).


#### - LLaVA-NeXT: A Strong Zero-shot Video Understanding Model
- [LLaVA-NeXT-Video](./docs/LLaVA-NeXT-Video.md): for video inference and evaluation scripts. We recommend to use [LMMs-video](https://lmms-lab.github.io/posts/lmms-eval-0.2/) for evaluation.

#### - LLaVA-NeXT: Tackling Multi-image, Video, and 3D in Large Multimodal Models
- [LLaVA-NeXT-Interleave](./docs/LLaVA-NeXT-Interleave.md): for multi-image demo and evaluation scripts.

## SGLang for SpeedUp Inference and Deployment

We use [SGLang](https://github.com/sgl-project/sglang) to speed up inference and deployment of LLaVA-NeXT. You could make LLaVA-NeXT as a backend API service with SGLang.

**Prepare Environment**:
    Following the instruction in the [sglang](https://github.com/sgl-project/sglang?tab=readme-ov-file#install)

### LLaVA-NeXT/OneVision

Checkout the HTTP Post/Get and SRT usage at [sglang/examples/runtime/llava_onevision](https://github.com/sgl-project/sglang/tree/main/examples/runtime/llava_onevision)

### LLaVA-NeXT (Video)

**Launch and Run on (K) Nodes**:
- Go to sglang project
    ```
    cd PATH_TO/sglang
    ```
- First node:
    ```sh
    bash examples/usage/llava_video/srt_example_llava_v.sh K 0 YOUR_VIDEO_PATH YOUR_MODEL_PATH FRAMES_PER_VIDEO
    (e.g. bash examples/usage/llava_video/srt_example_llava_v.sh K 0 examples/usage/llava_video/videos/Q98Z4OTh8RwmDonc.mp4 lmms-lab/LLaVA-NeXT-Video-7B-DPO 16)
    ```
- Second node:
    ```sh
    bash examples/usage/llava_video/srt_example_llava_v.sh K 1 YOUR_VIDEO_PATH YOUR_MODEL_PATH FRAMES_PER_VIDEO
    ```
- The K node:
    ```sh
    bash examples/usage/llava_video/srt_example_llava_v.sh K K-1 YOUR_VIDEO_PATH YOUR_MODEL_PATH FRAMES_PER_VIDEO
    ```


## Citation

If you find it useful for your research and applications, please cite related papers/blogs using this BibTeX:
```bibtex
@article{li2024llava,
  title={LLaVA-NeXT-Interleave: Tackling Multi-image, Video, and 3D in Large Multimodal Models},
  author={Li, Feng and Zhang, Renrui and Zhang, Hao and Zhang, Yuanhan and Li, Bo and Li, Wei and Ma, Zejun and Li, Chunyuan},
  journal={arXiv preprint arXiv:2407.07895},
  year={2024}
}

@misc{li2024llavanext-ablations,
	title={LLaVA-NeXT: What Else Influences Visual Instruction Tuning Beyond Data?},
	url={https://llava-vl.github.io/blog/2024-05-25-llava-next-ablations/},
	author={Li, Bo and Zhang, Hao and Zhang, Kaichen and Guo, Dong and Zhang, Yuanhan and Zhang, Renrui and Li, Feng and Liu, Ziwei and Li, Chunyuan},
	month={May},
	year={2024}
}

@misc{li2024llavanext-strong,
    title={LLaVA-NeXT: Stronger LLMs Supercharge Multimodal Capabilities in the Wild},
    url={https://llava-vl.github.io/blog/2024-05-10-llava-next-stronger-llms/},
    author={Li, Bo and Zhang, Kaichen and Zhang, Hao and Guo, Dong and Zhang, Renrui and Li, Feng and Zhang, Yuanhan and Liu, Ziwei and Li, Chunyuan},
    month={May},
    year={2024}
}

@misc{zhang2024llavanext-video,
  title={LLaVA-NeXT: A Strong Zero-shot Video Understanding Model},
  url={https://llava-vl.github.io/blog/2024-04-30-llava-next-video/},
  author={Zhang, Yuanhan and Li, Bo and Liu, haotian and Lee, Yong jae and Gui, Liangke and Fu, Di and Feng, Jiashi and Liu, Ziwei and Li, Chunyuan},
  month={April},
  year={2024}
}

@misc{liu2024llavanext,
    title={LLaVA-NeXT: Improved reasoning, OCR, and world knowledge},
    url={https://llava-vl.github.io/blog/2024-01-30-llava-next/},
    author={Liu, Haotian and Li, Chunyuan and Li, Yuheng and Li, Bo and Zhang, Yuanhan and Shen, Sheng and Lee, Yong Jae},
    month={January},
    year={2024}
}

@misc{liu2023improvedllava,
      title={Improved Baselines with Visual Instruction Tuning}, 
      author={Liu, Haotian and Li, Chunyuan and Li, Yuheng and Lee, Yong Jae},
      publisher={arXiv:2310.03744},
      year={2023},
}

@misc{liu2023llava,
      title={Visual Instruction Tuning}, 
      author={Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
      publisher={NeurIPS},
      year={2023},
}
```

## Acknowledgement

- [Vicuna](https://github.com/lm-sys/FastChat): the codebase we built upon, and our base model Vicuna-13B that has the amazing language capabilities!
- The LLaVA-NeXT project is currently maintained by the team along with our contributors (listed alphabetically by the first names): [Bo Li](https://brianboli.com/), [Dong Guo](https://www.linkedin.com/in/dongguoset/), [Feng Li](https://scholar.google.com/citations?hl=zh-CN&user=ybRe9GcAAAAJ&view_op=list_works&sortby=pubdate), [Hao Zhang](https://scholar.google.com/citations?user=B8hPxMQAAAAJ&hl=en), [Kaichen Zhang](https://www.linkedin.com/in/kaichen-zhang-014b17219/?originalSubdomain=sg), [Renrui Zhang](https://zrrskywalker.github.io/), [Yuanhan Zhang](https://zhangyuanhan-ai.github.io/), led by [Chunyuan Li](https://chunyuan.li/) and with the guidance and help from [Haotian Liu](https://hliu.cc/).
- The `﻿lmms-eval` framework and its core contributors, including Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, and Kairui Hu, for their support on the evaluation side.

## Related Projects

- [Instruction Tuning with GPT-4](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
- [LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day](https://github.com/microsoft/LLaVA-Med)
- [Otter: In-Context Multi-Modal Instruction Tuning](https://github.com/Luodian/Otter)

For future project ideas, please check out:
- [SEEM: Segment Everything Everywhere All at Once](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once)
- [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) to detect, segment, and generate anything by marrying [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) and [Segment-Anything](https://github.com/facebookresearch/segment-anything).
