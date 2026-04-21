import os
import sys
import csv
import zipfile
import argparse
import torch
import torchaudio
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.bidirectional_mamba import BidirMambaConfig, BidirMambaFlowNet
from model.dcae_wrapper import MusicDCAEWrapper
from model.conditioning import T5TextConditioner
from model.flow_generate import generate_music_flow

def enforce_10_seconds(wav_tensor: torch.Tensor, target_sr: int = 44100) -> torch.Tensor:
    """
    Strictly enforces the 10.0 second rule required by the ICME challenge.
    Pads with silence if too short, truncates if too long.
    """
    target_length = 10 * target_sr
    
    if wav_tensor.dim() == 1:
        wav_tensor = wav_tensor.unsqueeze(0) # Ensure (Channels, Time)
        
    current_length = wav_tensor.shape[1]
    
    if current_length > target_length:
        # Truncate
        return wav_tensor[:, :target_length]
    elif current_length < target_length:
        # Pad with zeros (silence)
        padding = target_length - current_length
        return torch.nn.functional.pad(wav_tensor, (0, padding))
    
    return wav_tensor

def main():
    parser = argparse.ArgumentParser(description="ICME 2026 Batch Generation")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to final_test_prompts.csv")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to your final model checkpoint")
    parser.add_argument("--team_name", type=str, required=True, help="Your official team name")
    parser.add_argument("--track", type=str, choices=["efficiency", "performance"], required=True)
    parser.add_argument("--sub_num", type=int, default=1, help="Submission number (1 or 2)")
    parser.add_argument("--out_dir", type=str, default="submission_wavs", help="Temp folder for wavs")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading checkpoint from {args.ckpt_path}...")
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model_config_dict = ckpt["model_config"]

    # 1. Initialize models
    mamba_cfg = BidirMambaConfig(**model_config_dict)
    model = BidirMambaFlowNet(mamba_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    conditioner = T5TextConditioner(d_model=mamba_cfg.conditioning_dim).to(device)
    conditioner.load_state_dict(ckpt.get("cond_state", {}), strict=False)
    conditioner.eval()
    
    print("Loading ACE-Step DCAE & Vocoder...")
    dcae = MusicDCAEWrapper(device=device)

    # 2. Read CSV Prompts
    # Assuming the CSV has columns like 'prompt_id' and 'prompt'
    # Adjust column names if the provided CSV differs
    prompts = []
    with open(args.csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            p_id = row.get('id')
            p_text = row.get('prompt')
            prompts.append((p_id, p_text))

    print(f"Found {len(prompts)} prompts. Starting batch generation...")

    # 3. Generate Audio
    generated_files = []
    for i, (p_id, p_text) in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] Generating: {p_id} -> '{p_text[:50]}...'")
        
        # Call your generation function
        # Ensure you pass in the correct loaded instances
        wav_tensor = generate_music_flow(
            prompt=p_text,
            model=model,
            conditioner=conditioner, # Replace with your actual variable
            dcae=dcae,
            duration_sec=10.5,
            n_steps=100,
            device=device
        )
        
        wav_tensor = enforce_10_seconds(wav_tensor, target_sr=44100)
        
        # Save to disk
        out_path = os.path.join(args.out_dir, f"{p_id}.wav")
        torchaudio.save(out_path, wav_tensor, sample_rate=44100)
        generated_files.append(out_path)

    # 4. Package for Submission
    zip_filename = f"{args.team_name}_{args.track}_{args.sub_num}.zip"
    print(f"Zipping {len(generated_files)} files into {zip_filename}...")
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in generated_files:
            # Write just the filename inside the zip, not the folder structure
            zipf.write(file, arcname=os.path.basename(file))
            
    print(f"Success! Ready to email {zip_filename} to andrew891221@gmail.com")

if __name__ == "__main__":
    main()