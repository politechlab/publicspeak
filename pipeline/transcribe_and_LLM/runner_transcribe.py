import subprocess
import os
import time
import json

j = 0

os.environ["CUDA_VISIBLE_DEVICES"] = '7'

cities = ["03000"]

for city in cities:
    for i in os.listdir(f"/home/shared/turbo_data/localgov/videos/2022/{city}"):
        os.makedirs(f'../../data/transcripts/{city}', exist_ok=True)
        if i.endswith("mp4") and (i[:-3]+"json" not in os.listdir(f"../../data/transcripts/{city}")):
            command = f"python main.py --mode to_wav --wav_input '/home/shared/turbo_data/localgov/videos/2022/{city}/{i}' --wav_output '../../data/audio'"
            subprocess.run(command, shell=True, check=True)
            command = f"python main.py --mode transcribe --device cuda --ts_output_folder ../../data/transcripts/{city} --audio_file '../../data/audio/{i[:-3]+'wav'}'"
            subprocess.run(command, shell=True, check=True)
            json_name = i[:-3]+"json"
