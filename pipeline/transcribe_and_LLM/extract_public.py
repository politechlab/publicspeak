import json
import openai
from collections import defaultdict
import os
from constant import keyword_list

def count_long_text_ratio(clean_data, long_text_th=50):
    
    # Count the merged utterances for each speaker
    merged_long_utterance_counts = defaultdict(int)
    merged_utterance_counts = defaultdict(int)
    for entry in clean_data:
        merged_utterance_counts[entry['speaker']] += 1
        if len(entry["text"].split()) > long_text_th:
            merged_long_utterance_counts[entry['speaker']] += 1
    temp_dict = {}
    for key in merged_utterance_counts:
        if key not in merged_long_utterance_counts:
            temp_dict[key] = 0
        else:
            temp_dict[key] = merged_long_utterance_counts[key]/merged_utterance_counts[key]
    return temp_dict

def extract_public(args, this_merged, this_mapping, this_triggers):
    ratio_list = count_long_text_ratio(this_merged, args.long_text_th)
    pub_comments = {}
    se_pair = []
    for i in this_triggers:
        try:
            st = this_mapping[str(this_triggers[i]["start"]["numbering"]) + ". "]
            ed = this_mapping[str(this_triggers[i]["end"]["numbering"]) + ". "]
            pub_comments[i] = this_merged[st: ed]
            se_pair.append((st, ed))
        except:
            st = 0
            ed = 0
            se_pair.append([st, ed])
    
    outside_speaker = set()
    for i in range(len(this_merged)):
        isin = False
        for st, ed in se_pair:
            isin = isin or st < i < ed
        if not isin:
            outside_speaker.add(this_merged[i]["speaker"])

    public = {}
    if pub_comments:
        for key, item in pub_comments.items():
            public[key] = []
            for i in item:
                if i["speaker"] not in outside_speaker and ratio_list[i["speaker"]] >= args.ratio_count:
                    #print({"start": i["start"], "speaker": i["speaker"], "text": i["text"]})
                    public[key].append({"start": i["start"], "speaker": i["speaker"], "text": i["text"]})
    
    return public

def clean_and_find_manager(path):
    with open(path) as file:
        transcript_data = json.load(file)
    
    
    
    hallucination_indicators = [
        "https", "http", ".com", ".ai", ".org", ".net", "algorithm", 
        "neural", "whisper", "openai", "machine learning", "training data", "dataset",
        "transcription", "accuracy", "deep learning", "tensor", "compute", "gpu", 
        "castingwords"
    ]


    clean_data = []
    for utterance in transcript_data['segments']:
        utterance["text"] = str(utterance["text"])
        if not any(indicator in utterance["text"].lower() for indicator in hallucination_indicators) and len(utterance["text"]) > 0:
            if "speaker" not in utterance:
                utterance["speaker"] = "UNKNOWN"
            clean_data.append(utterance)
    
    merged_data = []
    current_speaker = None
    current_text = ""
    current_end = 0
    for entry in clean_data:
        if current_speaker is None:
            current_speaker = entry['speaker']
            current_text = entry['text']
            temp_start = entry['start']
            current_end = entry['end']
        elif current_speaker == entry['speaker'] and entry["start"] - current_end < 10:
            current_text += " " + entry['text']
            current_end = entry['end']
        else:
            merged_data.append({"start": temp_start, "end": current_end, "speaker": current_speaker, "text": current_text})
            current_speaker = entry['speaker']
            current_text = entry['text']
            temp_start = entry['start']
            current_end = entry['end']
    # Add the last entry
    if current_text:
        merged_data.append({"start": temp_start, "end": clean_data[-1]["end"], "speaker": current_speaker, "text": current_text})
    
    #merged_data = clean_data
    merged_adjacent_utterances = merged_data
    
    # Count the merged utterances for each speaker
    merged_utterance_counts = defaultdict(int)
    for entry in merged_adjacent_utterances:
        merged_utterance_counts[entry['speaker']] += 1
    
    # Identify the speaker with the most merged utterances for this file
    top_3_keys = [k for k, v in sorted(merged_utterance_counts.items(), key=lambda item: item[1], reverse=True)[:3]]
    for item in clean_data:
        if item['speaker'] in top_3_keys:
            if "Pledge of Allegiance" in item['text'] or "roll call" in item['text']:
                return item['speaker'], merged_data
    key = top_3_keys[0]
    return key, merged_data


def cut_off(content, args):
    total_list = content.split("\n")
    temp_list = []
    for i in total_list:
        # if len(i.split()) < 5:
        #     continue
        if len(i.split()) >= args.cut_off_th:
            sentences = i.split(".")
            chunks = []
            chunk_len = 0
            chunk = ""
            for j in sentences:
                chunk += j + " "
                chunk_len += len(j.split())
                if chunk_len > args.cut_off_th:
                    chunks.append(chunk)
                    chunk_len = 0
                    chunk = ""
            
            
            out_str = ""
            for c in chunks:
                status = 0
                for kw in keyword_list:
                    if kw in c:
                        status = 1
                if status:
                    out_str += c + " "
            if out_str:
                temp_list.append(out_str)
        else:
            temp_list.append(i) 
    return "\n".join(temp_list)

def ask_gpt(initial_prompt, content, model="gpt-4", temperature=0):
    print(model)
    openai.api_key = os.getenv("OPENAI_API_KEY") 
    msg = initial_prompt + content
    messages = [
        {"role": "user", "content": msg}
    ]

    response = openai.ChatCompletion.create(
      model=model,
      messages=messages,
      temperature=temperature
    )
    # print("#####################")
    # print(response.choices[0].message.content)
    try:
        return eval(response.choices[0].message.content)
    except:
        return {}
    
def ask_gpt_multi(initial_prompt, content, the_json, model="gpt-4", temperature=0):
    openai.api_key = os.getenv("OPENAI_API_KEY") 
    msg = "The JSON file: \n" + json.dumps(the_json) + "\n Transcript segment: \n" + content
    
    messages = [
        {"role": "system", "content": initial_prompt},
        {"role": "user", "content": msg}
    ]

    response = openai.ChatCompletion.create(
      model=model,
      messages=messages,
      temperature=temperature,
      #response_format={"type": "json_object"},
    )
    
    # print(response.choices[0].message.content)
    # print("#####################")
    try:
        return eval(response.choices[0].message.content)
    except:
        return {}
    
    
def ask_gpt_new(initial_prompt, content, model="gpt-4o", temperature=0):
    print(model)
    openai.api_key = os.getenv("OPENAI_API_KEY") 
    msg = "\n Transcript segment: \n" + content
    
    messages = [
        {"role": "system", "content": initial_prompt},
        {"role": "user", "content": msg}
    ]

    response = openai.ChatCompletion.create(
      model=model,
      messages=messages,
      temperature=temperature
    )
    # print("#####################")
    # print(response.choices[0].message.content)
    print( response.choices[0].message.content[7:-4].strip())
    print("```json" in response.choices[0].message.content)
    if "```json" in response.choices[0].message.content:
        
        c = response.choices[0].message.content[7:-4].strip()
    try:
        return eval(c)
    except:
        return {}