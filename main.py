import torch
import argparse
from TTS.api import TTS

# Argument Parser Setup
parser = argparse.ArgumentParser(description='Coqui AI TTS')
parser.add_argument('-t', '--text', required=True, help='Text to have spoken by model')
args = parser.parse_args()

# Get device (for models that might use CUDA, though TTS might not need it)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize TTS model (Coqui TTS doesn't require .to(device) as it doesn't use PyTorch for this)
tts = TTS("tts_models/en/ljspeech/fast_pitch")

# Text to speech to a file
tts.tts_to_file(text=args.text, file_path="output.wav")

print(f"Speech synthesis completed, output saved to 'output.wav'.")

