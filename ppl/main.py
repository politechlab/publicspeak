from helper import download_youtube_video, convert_to_wav, video_filename, clean_text
from extract_public import clean_and_find_manager, cut_off, ask_gpt, extract_public, ask_gpt_multi, ask_gpt_new
#from extract_public import ask_gpt_multi
from prompt_factory import initial_prompt, initial_prompt_multi, initial_prompt_general, initial_prompt_PC
from transcribe import use_whisperx
from options import args
import os
import prompt_factory
from pathlib import Path
import json
from collections import defaultdict
import openai
import time

# def ask_gpt_multi(initial_prompt, content, the_json, model="gpt-4", temperature=0):
#     openai.api_key = os.getenv("OPENAI_API_KEY") 
#     msg = "The JSON file: \n" + json.dumps(the_json) + "\n Transcript segment: \n" + content
    
#     messages = [
#         {"role": "system", "content": initial_prompt},
#         {"role": "user", "content": msg}
#     ]

#     response = openai.ChatCompletion.create(
#       model=model,
#       messages=messages,
#       temperature=temperature,
#       #response_format={"type": "json_object"},
#     )
    
#     # print(response.choices[0].message.content)
#     # print("#####################")
#     try:
#         return eval(response.choices[0].message.content)
#     except:
#         return {}

    
if __name__ == "__main__":
    
    if args.mode == "full":
        pass
    
    elif args.mode == "download_video":
        output_path = args.video_output  # Replace with your desired path
        url = args.url
        # TODO: change to Path
        
        # if os.path.exists(output_path + ""):
        #     print(f"File {output_path} is existing, skip the downloading process.")
        # else:
        #     if not os.path.exists(seg_dir):
        #         os.makedirs(seg_dir)
        #         print(f"File {output_path} does not exist.")
        download_youtube_video(url, output_path)
    elif args.mode == "to_wav":
        input_file = args.wav_input
        # TODO: change wav_output
        # output_path = args.wav_output
        convert_to_wav(input_file, "audio_2023/" + input_file.split("/")[-1][:-4] + ".wav")
    elif args.mode == "transcribe":
        model_name = args.model_name
        device = args.device
        hf_token = os.getenv("HF_TOKEN")
        audio_file = args.audio_file
        batch_size = args.batch_size
        compute_type = args.compute_type
        ts_output_folder = args.ts_output_folder
        try:
            result = use_whisperx(device, audio_file, hf_token, batch_size, compute_type, model_name)
            json_name = os.path.splitext(audio_file)[0].split("/")[-1] + ".json"

            if not os.path.exists(ts_output_folder):
                os.makedirs(ts_output_folder)
            with open(ts_output_folder + "/" + json_name, 'w') as f:
                json.dump(result, f)
        except KeyError:
            print(f"No word is unknown.")
 #TODO: Remove the code above the line, and put them into other scripts.           
    elif args.mode == "find_public_trigger":
        ts_path = args.ts_path
        
        temp = []
        result = {}
        
        if ts_path.endswith("json"):
            manager, merged_data = clean_and_find_manager(ts_path)
            cnt = 1
            idx_mapping = {}
            content = ""
            result["merged_data"] = merged_data
            for i, item in enumerate(merged_data):
                if item["speaker"] == manager:
                    content += str(cnt) + ". " + item["text"].strip() + "\n"
                    idx_mapping[str(cnt) + ". "] = i
                    cnt += 1
            result["idx_map"] = idx_mapping
            ################################
            # Add this if you use GPT4
            content = cut_off(content, args)
            gpt_model = args.gpt_version
            #gpt_model = "gpt-4o"
            # result["triggers"] = ask_gpt(initial_prompt = initial_prompt, model=gpt_model, content=content)
            result["triggers"] = ask_gpt(initial_prompt = initial_prompt, model=gpt_model, content=content)
            trigger_path = os.path.splitext(ts_path)[0] + "_trigger.json"
            with open(trigger_path, "w") as f:
                json.dump(result, f)

    elif args.mode == "find_public_trigger_new":
        ts_path = args.ts_path
        
        temp = []
        result = {}
        
        if ts_path.endswith("json"):
            manager, merged_data = clean_and_find_manager(ts_path)
            cnt = 1
            idx_mapping = {}
            content = ""
            result["merged_data"] = merged_data
            for i, item in enumerate(merged_data):
                content += str(cnt) + ". " + item["text"].strip() + "\n"
                idx_mapping[str(cnt) + ". "] = i
                cnt += 1
            result["idx_map"] = idx_mapping
            ################################
            # Add this if you use GPT4
            #content = cut_off(content, args)
            gpt_model = args.gpt_version
            gpt_model = "gpt-4o"
            # result["triggers"] = ask_gpt(initial_prompt = initial_prompt, model=gpt_model, content=content)
            result["triggers"] = ask_gpt_new(initial_prompt = initial_prompt, model=gpt_model, content=content)
            trigger_path = os.path.splitext(ts_path)[0] + "_trigger.json"
            with open(trigger_path, "w") as f:
                json.dump(result, f)            
    
    elif args.mode == "find_public_trigger_multi":
        ts_path = args.ts_path
        manager, merged_data = clean_and_find_manager(ts_path)
        cnt = 0
        idx_mapping = {}
        content = ""
        result = {}
        result["merged_data"] = merged_data
        the_json = {}
        length = 0
        total_json = {}
        gpt_model = args.gpt_version
        for i, item in enumerate(merged_data):
            content += str(cnt) + ". " + item["text"].strip() + "\n"
            idx_mapping[str(cnt) + ". "] = i
            cnt += 1
            length += len((str(cnt) + ". " + item["text"].strip() + "\n").split())
            if length > 3500:
                new_json = ask_gpt_multi(initial_prompt = initial_prompt_multi, the_json = the_json,model=gpt_model, content=content)
                if not new_json:
                    new_json = ask_gpt_multi(initial_prompt = initial_prompt_multi, the_json = the_json,model=gpt_model, content=content, temperature=0.5)
                if not new_json:
                    new_json = ask_gpt_multi(initial_prompt = initial_prompt_multi, the_json = the_json,model=gpt_model, content=content, temperature=1)
                if new_json:
                    the_json = new_json
                    for key in new_json:
                        if key not in total_json:
                            total_json[key] = new_json[key]

                content = ""
                length = 0
        if content:
            new_json = ask_gpt_multi(initial_prompt = initial_prompt_multi, the_json = the_json,model=gpt_model, content=content)
            if not new_json:
                new_json = ask_gpt_multi(initial_prompt = initial_prompt_multi, the_json = the_json,model=gpt_model, content=content, temperature=0.5)
            if not new_json:
                new_json = ask_gpt_multi(initial_prompt = initial_prompt_multi, the_json = the_json,model=gpt_model, content=content, temperature=1)
            if new_json:
                the_json = new_json
                for key in new_json:
                    if key not in total_json:
                        total_json[key] = new_json[key]

        result["idx_map"] = idx_mapping
        result["triggers"] = total_json
        trigger_path = os.path.splitext(ts_path)[0] + "_trigger_multi.json"
        with open(trigger_path, "w") as f:
            json.dump(result, f)
    
    elif args.mode == "find_public_trigger_general":
        ts_path = args.ts_path
        manager, merged_data = clean_and_find_manager(ts_path)
        cnt = 0
        idx_mapping = {}
        content = ""
        result = {}
        result["merged_data"] = merged_data
        the_json = {}
        length = 0
        total_json = {}
        gpt_model = args.gpt_version
        for i, item in enumerate(merged_data):
            content += str(cnt) + ". " + item["text"].strip() + "\n"
            idx_mapping[str(cnt) + ". "] = i
            cnt += 1
            length += len((str(cnt) + ". " + item["text"].strip() + "\n").split())
            if length > 3500:
                new_json = ask_gpt_multi(initial_prompt = initial_prompt_general, the_json = the_json,model=gpt_model, content=content)
                if not new_json:
                    new_json = ask_gpt_multi(initial_prompt = initial_prompt_general, the_json = the_json,model=gpt_model, content=content, temperature=0.5)
                if not new_json:
                    new_json = ask_gpt_multi(initial_prompt = initial_prompt_general, the_json = the_json,model=gpt_model, content=content, temperature=1)
                if new_json:
                    the_json = new_json
                    for key in new_json:
                        if key not in total_json:
                            total_json[key] = new_json[key]

                content = ""
                length = 0
        if content:
            new_json = ask_gpt_multi(initial_prompt = initial_prompt_general, the_json = the_json,model=gpt_model, content=content)
            if not new_json:
                new_json = ask_gpt_multi(initial_prompt = initial_prompt_general, the_json = the_json,model=gpt_model, content=content, temperature=0.5)
            if not new_json:
                new_json = ask_gpt_multi(initial_prompt = initial_prompt_general, the_json = the_json,model=gpt_model, content=content, temperature=1)
            if new_json:
                the_json = new_json
                for key in new_json:
                    if key not in total_json:
                        total_json[key] = new_json[key]

        result["idx_map"] = idx_mapping
        result["triggers"] = total_json
        trigger_path = os.path.splitext(ts_path)[0] + "_trigger_general.json"
        with open(trigger_path, "w") as f:
            json.dump(result, f)
    
    elif args.mode == "find_public_trigger_PC":
        ts_path = args.ts_path
        manager, merged_data = clean_and_find_manager(ts_path)
        cnt = 0
        idx_mapping = {}
        content = ""
        result = {}
        result["merged_data"] = merged_data
        the_json = {}
        length = 0
        total_json = {}
        gpt_model = args.gpt_version
        for i, item in enumerate(merged_data):
            content += str(cnt) + ". " + item["text"].strip() + "\n"
            idx_mapping[str(cnt) + ". "] = i
            cnt += 1
            length += len((str(cnt) + ". " + item["text"].strip() + "\n").split())
            if length > 3500:
                new_json = ask_gpt_multi(initial_prompt = initial_prompt_PC, the_json = the_json,model=gpt_model, content=content)
                if not new_json:
                    new_json = ask_gpt_multi(initial_prompt = initial_prompt_PC, the_json = the_json,model=gpt_model, content=content, temperature=0.5)
                if not new_json:
                    new_json = ask_gpt_multi(initial_prompt = initial_prompt_PC, the_json = the_json,model=gpt_model, content=content, temperature=1)
                if new_json:
                    the_json = new_json
                    for key in new_json:
                        if key not in total_json:
                            total_json[key] = new_json[key]

                content = ""
                length = 0
        if content:
            new_json = ask_gpt_multi(initial_prompt = initial_prompt_PC, the_json = the_json,model=gpt_model, content=content)
            if not new_json:
                new_json = ask_gpt_multi(initial_prompt = initial_prompt_PC, the_json = the_json,model=gpt_model, content=content, temperature=0.5)
            if not new_json:
                new_json = ask_gpt_multi(initial_prompt = initial_prompt_PC, the_json = the_json,model=gpt_model, content=content, temperature=1)
            if new_json:
                the_json = new_json
                for key in new_json:
                    if key not in total_json:
                        total_json[key] = new_json[key]

        result["idx_map"] = idx_mapping
        result["triggers"] = total_json
        trigger_path = os.path.splitext(ts_path)[0] + "_trigger_PC.json"
        with open(trigger_path, "w") as f:
            json.dump(result, f)
    
    elif args.mode == "public_extraction":
        ts_path = args.ts_path
        trigger_path = os.path.splitext(ts_path)[0] + "_trigger.json"
        with open(trigger_path, "r") as f:
            triggers = json.load(f)
        this_merged = triggers["merged_data"]
        this_mapping = triggers["idx_map"]
        this_triggers = triggers["triggers"]
        public = extract_public(args, this_merged, this_mapping, this_triggers)
        public_path = os.path.splitext(ts_path)[0] + "_public.json"
        with open(public_path, "w") as f:
            json.dump(public, f)
            
    elif args.mode == "public_extraction_multi":
        # extract without identifying transitions
        ts_path = args.ts_path
        trigger_path = os.path.splitext(ts_path)[0] + "_trigger_multi.json"
        with open(trigger_path, "r") as f:
            triggers = json.load(f)
        this_merged = triggers["merged_data"]
        this_mapping = triggers["idx_map"]
        this_triggers = triggers["triggers"]
        public = extract_public(args, this_merged, this_mapping, this_triggers)
        public_path = os.path.splitext(ts_path)[0] + "_public_multi.json"
        with open(public_path, "w") as f:
            json.dump(public, f)
    
    elif args.mode == "public_extraction_general":
        # extract without identifying transitions
        ts_path = args.ts_path
        trigger_path = os.path.splitext(ts_path)[0] + "_trigger_general.json"
        with open(trigger_path, "r") as f:
            triggers = json.load(f)
        this_merged = triggers["merged_data"]
        this_mapping = triggers["idx_map"]
        this_triggers = triggers["triggers"]
        public = extract_public(args, this_merged, this_mapping, this_triggers)
        public_path = os.path.splitext(ts_path)[0] + "_public_general.json"
        with open(public_path, "w") as f:
            json.dump(public, f)

    elif args.mode == "public_extraction_PC":
        # extract without identifying transitions
        ts_path = args.ts_path
        trigger_path = os.path.splitext(ts_path)[0] + "_trigger_PC.json"
        with open(trigger_path, "r") as f:
            triggers = json.load(f)
        this_merged = triggers["merged_data"]
        this_mapping = triggers["idx_map"]
        this_triggers = triggers["triggers"]
        public = extract_public(args, this_merged, this_mapping, this_triggers)
        public_path = os.path.splitext(ts_path)[0] + "_public_PC.json"
        with open(public_path, "w") as f:
            json.dump(public, f)
        