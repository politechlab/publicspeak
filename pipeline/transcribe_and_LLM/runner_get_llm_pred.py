import subprocess
import os
import time
import json
current_dir = os.path.dirname(os.path.realpath(__file__))
out_dir = os.path.abspath(os.path.join(current_dir, "../../data/LLM_indicators/"))
data_dir = os.path.abspath(os.path.join(current_dir, "../../data/"))
def gen_trigger(name):
    
    
     
    with open(name) as f:
        js = json.load(f)
    for key in js:

        json_name = key + ".json"
        with open(os.path.join(out_dir, json_name), "w") as f:
            json.dump({"segments": js[key]}, f)

        command = f"python {current_dir}/main.py --mode find_public_trigger_general --ts_path '{out_dir}/{json_name}'"
        subprocess.run(command, shell=True, check=True)
        command = f"python {current_dir}/main.py --mode public_extraction_general --ts_path '{out_dir}/{json_name}'"
        subprocess.run(command, shell=True, check=True)

cities = ["AA"]

for c in cities:
    gen_trigger(f"{data_dir}/raw_train/{c}_train.json")
    gen_trigger(f"{data_dir}/raw_val/{c}_val.json")
    gen_trigger(f"{data_dir}/raw_test/{c}_test.json")
