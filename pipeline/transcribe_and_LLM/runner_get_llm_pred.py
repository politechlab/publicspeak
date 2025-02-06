import subprocess
import os
import time
import json

def gen_trigger(name):
    trigger = 1
     
    with open(name) as f:
        js = json.load(f)
    for key in js:
        if trigger:
            json_name = key + ".json"
            with open('../../data/LLM_indicators/' + json_name, "w") as f:
                json.dump({"segments": js[key]}, f)

            command = f"python main.py --mode find_public_trigger_general --ts_path '../../data/LLM_indicators/{json_name}'"
            subprocess.run(command, shell=True, check=True)
            command = f"python main.py --mode public_extraction_general --ts_path '../../data/LLM_indicators/{json_name}'"
            subprocess.run(command, shell=True, check=True)

cities = ["AA"]

for c in cities:
    gen_trigger(f"../../data/raw_train/{c}_train.json")
    gen_trigger(f"../../data/raw_val/{c}_val.json")
    gen_trigger(f"../../data/raw_test/{c}_test.json")
