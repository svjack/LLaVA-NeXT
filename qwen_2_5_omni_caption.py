import os
import argparse
import soundfile as sf
from tqdm import tqdm
from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

def main(args):
    # Initialize the model and processor
    model = Qwen2_5OmniModel.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B",
        torch_dtype="auto",
        device_map="auto",
        enable_audio_output=False  # We only need text output for captioning
    )
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)

    # Process each video file in the input directory
    for video_file in tqdm(os.listdir(args.input_path), desc="Processing videos"):
        if video_file.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
            video_path = os.path.join(args.input_path, video_file)
            
            # Prepare the conversation with the video
            conversation = [
                {
                    "role": "system",
                    "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video_path},
                        {"type": "text", "text": "Please describe this video in detail."}
                    ],
                }
            ]

            # Prepare inputs for the model
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=args.use_audio)
            inputs = processor(text=text, audios=audios, images=images, videos=videos, 
                             return_tensors="pt", padding=True)
            inputs = inputs.to(model.device).to(model.dtype)

            # Generate description
            text_ids = model.generate(**inputs, use_audio_in_video=args.use_audio, return_audio=False)
            description = processor.batch_decode(text_ids, skip_special_tokens=True, 
                                              clean_up_tokenization_spaces=False)[0]

            # Save the description to a text file
            output_file = os.path.splitext(video_file)[0] + ".txt"
            output_path = os.path.join(args.output_path, output_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(description)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video captioning using Qwen2.5-Omni model")
    parser.add_argument('--input_path', type=str, required=True, 
                       help='Path to directory containing input videos')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to directory where captions will be saved')
    parser.add_argument('--use_audio', action='store_true',
                       help='Whether to use audio from videos (if present)')
    args = parser.parse_args()
    
    main(args)
