import os
import json
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter
import functools
import random
import numpy as np
from tqdm import tqdm
import sys
import argparse
from typing import Dict, List, Any, Tuple

def load_data(city: str, data_dir: str) -> Tuple[Dict[str, Any], int, int]:
    """
    加载训练、验证和测试数据
    
    Args:
        city: 城市名称
        data_dir: 数据目录
        
    Returns:
        Tuple[Dict[str, Any], int, int]: 合并后的数据字典，训练集大小，验证集大小
    """
    with open(os.path.join(data_dir, f"raw_train/{city}_train.json")) as f:
        train = json.load(f)
    with open(os.path.join(data_dir, f"raw_val/{city}_val.json")) as f:
        val = json.load(f)
    with open(os.path.join(data_dir, f"raw_test/{city}_test.json")) as f:
        test = json.load(f)

    truth = {}
    for key in train:
        truth[key] = train[key]
    train_limit = len(train.keys())
    for key in val:
        truth[key] = val[key]
    test_limit = train_limit + len(val.keys())
    for key in test:
        truth[key] = test[key]
        
    # 处理文本格式
    for item in truth:
        for m in range(len(truth[item])):
            truth[item][m]["text"] = str(truth[item][m]["text"])
            
    return truth, train_limit, test_limit

def process_section_types(truth: Dict[str, Any], current_dir: str) -> None:
    """
    处理每个会议的部分类型
    
    Args:
        truth: 数据字典
        current_dir: 当前目录
    """
    for j in truth:
        with open(os.path.join(current_dir, f"../../data/LLM_indicators/{j}_trigger_general.json")) as f:
            triggers = json.load(f)

        this_mapping = triggers["idx_map"]
        this_triggers = triggers["triggers"]
        comment_st_pl = []
        comment_ed_pl = []
        hearing_st_pl = []
        hearing_ed_pl = []
        pc_pair = []
        ph_pair = []
        
        for i in this_triggers:
            try:
                st = this_mapping[str(this_triggers[i]["start"]["numbering"]) + ". "]
                ed = this_mapping[str(this_triggers[i]["end"]["numbering"]) + ". "]
                if "comment" in i:
                    pc_pair.append((st, ed))
                else:
                    ph_pair.append((st, ed))
            except:
                st = -1
                ed = -1
                pc_pair.append([st, ed])
                ph_pair.append([st, ed])
        
        for i in range(len(truth[j])):
            truth[j][i]["section_type_gpt"] = "other"
            isin = False
            for st, ed in ph_pair:
                isin = isin or st <= i <= ed
            if isin:
                truth[j][i]["section_type_gpt"] = "public_hearing"
                
            isin = False
            for st, ed in pc_pair:
                isin = isin or st <= i <= ed
            if isin:
                truth[j][i]["section_type_gpt"] = "public_comment"

def generate_processed_data(city: str, 
                          data_dir: str,
                          output_dir: str,
                          plm_file_name: str = "AA_pred_LOO_roberta.json",
                          seed: int = 42) -> None:
    """
    生成处理后的数据
    
    Args:
        city: 城市名称
        data_dir: 数据目录
        output_dir: 输出目录
        plm_file_name: PLM预测文件名
        seed: 随机种子
    """
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    
    # 获取当前目录
    current_dir = os.path.dirname(os.path.realpath(__file__))
    
    # 加载数据
    truth, train_limit, test_limit = load_data(city, data_dir)
    
    # 处理部分类型
    process_section_types(truth, current_dir)
    
    # 准备数据
    data = [truth[i] for i in truth]
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    subdir_path = os.path.join(output_dir, "mapping")
    if not os.path.exists(subdir_path):
        os.makedirs(subdir_path)
        
    # 保存映射
    with open(os.path.join(subdir_path, "all_id_mapping.json"), "w") as f:
        json.dump({"i2u": i2t, "u2i": t2i, "i2m": i2m, "m2i": m2i, "info2t": info2t, "t2info": t2info}, f)
    
    # 创建子目录
    for subdir in ["train", "eval", "test"]:
        subdir_path = os.path.join(output_dir, subdir)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)
            
    # 分割数据集
    training_set = data[:train_limit]
    test_set = data[train_limit: test_limit]
    testv_set = data[test_limit:]
    
    # 处理数据
    training_set = sum(training_set, [])
    test_set = sum(test_set, [])
    testv_set = sum(testv_set, [])
    
    # 设置PLM文件路径
    plm_location = os.path.join(current_dir, "../../data/PLM_indicators")
    plm_path = os.path.join(plm_location, plm_file_name)
    
    # 运行数据处理
    run_through(training_set, test_set, testv_set, output=output_dir, plm_path=plm_path)

def run_through(train_list, test_list, testv_list, output="../data/public_comments", plm_path=None):
    """
    运行数据处理
    
    Args:
        train_list: 训练数据列表
        test_list: 测试数据列表
        testv_list: 验证数据列表
        output: 输出目录
        plm_path: PLM预测文件路径
    """
    contains_train, contains_test, contains_testv, vocab = make_contain(train_list, test_list, testv_list)
    
    hearing_signal = ["open up the public hearing","public hearing"]
    comment_signal = ["public comment"]
    next_signal = ["next speaker"]
    name_signal = ["my name is"]
    
    train_commenttype = []
    train_commenttype_tar = []
    train_llm_commenttype = []
    train_db_commenttype = []
    
    train_speaker_type_dict = {}
    train_speaker_type = []
    train_speaker_type_tar = []
    
    train_speaker_long_dict = {}
    train_speaker_long = []
    train_speaker_count_dict = {}
    spoken_train = []
    longUtter_train = []
    
    train_section_type_gpt = []
    train_section_type = []
    train_section_type_tar = []
    train_precede = []
    
    train_meeting_now = ""
    train_first = []
    
    train_hearing_phrase = []
    train_comment_phrase = []
    train_next_phrase = []
    train_name_phrase = []

    # 加载PLM预测结果
    with open(plm_path) as f:
        plm_pred = json.load(f)

    for ind, example in enumerate(train_list):
        spoken_train.append(make_spoken(example))
        train_commenttype += make_commenttype(example)
        train_commenttype_tar += make_commenttype_tar(example)
        train_llm_commenttype.append(make_plm_pred(example, ind, plm_pred["train_pred"]))
        
        train_section_type_gpt.extend(make_section_type_gpt(example))
        train_section_type.extend(make_section_type(example))
        train_section_type_tar.extend(make_section_type_tar(example))
        
        train_speaker_type_dict = _make_speakertype(example, train_speaker_type_dict)
        longU, train_speaker_long_dict = make_longUtter(example, train_speaker_long_dict)
        longUtter_train.append(longU)
        
        precede = make_precede(ind, train_list)
        if precede:
            train_precede.append(precede)
        first, train_meeting_now = make_first(train_meeting_now, example)
        if first:
            train_first.append(first)
            
        train_speaker_count_dict = get_speak_count(example, train_speaker_count_dict)
        
        train_hearing_phrase.append(make_phrase(example, hearing_signal))
        train_comment_phrase.append(make_phrase(example, comment_signal))
        train_next_phrase.append(make_phrase(example, next_signal))
        train_name_phrase.append(make_phrase(example, name_signal))
        
    train_speaker_long = make_long_utter_rate(train_speaker_long_dict)
    train_speaker_type = make_speakertype(train_speaker_type_dict)
    for i in range(0, len(train_speaker_type), 2):
        train_speaker_type_tar += make_speaker_type_tar(train_speaker_type[i]) 
    
    train_speaker_low_count = make_low_count(train_speaker_count_dict)
    train_speaker_high_count = make_high_count(train_speaker_count_dict)
    
    test_commenttype = []
    test_commenttype_tar = []
    test_llm_commenttype = []
    test_db_commenttype = []
    
    test_speaker_type_dict = {}
    test_speaker_type = []
    test_speaker_type_tar = []
    
    test_speaker_long_dict = {}
    test_speaker_long = []
    test_speaker_count_dict = {}
    spoken_test = []
    longUtter_test = []
    
    test_section_type_gpt = []
    test_section_type = []
    test_section_type_tar = []
    test_precede = []
    
    test_meeting_now = ""
    test_first = []
    
    test_hearing_phrase = []
    test_comment_phrase = []
    test_next_phrase = []
    test_name_phrase = []
    
    for ind, example in enumerate(test_list):
        spoken_test.append(make_spoken(example))
        test_commenttype += make_commenttype(example)
        test_commenttype_tar += make_commenttype_tar(example)
        test_llm_commenttype.append(make_plm_pred(example, ind, plm_pred["val_pred"]))
        
        test_section_type_gpt.extend(make_section_type_gpt(example))
        test_section_type.extend(make_section_type(example))
        test_section_type_tar.extend(make_section_type_tar(example))
        
        test_speaker_type_dict = _make_speakertype(example, test_speaker_type_dict)
        longU, test_speaker_long_dict = make_longUtter(example, test_speaker_long_dict)
        longUtter_test.append(longU)
        
        precede = make_precede(ind, test_list)
        if precede:
            test_precede.append(precede)
        first, test_meeting_now = make_first(test_meeting_now, example)
        if first:
            test_first.append(first)
           
        test_speaker_count_dict = get_speak_count(example, test_speaker_count_dict)
        
        test_hearing_phrase.append(make_phrase(example, hearing_signal))
        test_comment_phrase.append(make_phrase(example, comment_signal))
        test_next_phrase.append(make_phrase(example, next_signal))
        test_name_phrase.append(make_phrase(example, name_signal))
    
    test_speaker_long = make_long_utter_rate(test_speaker_long_dict)
    
    test_speaker_type = make_speakertype(test_speaker_type_dict)
    for i in range(0, len(test_speaker_type), 2):
        test_speaker_type_tar += make_speaker_type_tar(test_speaker_type[i])
    
    test_speaker_low_count = make_low_count(test_speaker_count_dict)
    test_speaker_high_count = make_high_count(test_speaker_count_dict)
    
    
    testv_commenttype = []
    testv_commenttype_tar = []
    testv_llm_commenttype = []
    testv_db_commenttype = []
    
    testv_speaker_type_dict = {}
    testv_speaker_type = []
    testv_speaker_type_tar = []
    
    testv_speaker_long_dict = {}
    testv_speaker_long = []
    testv_speaker_count_dict = {}
    spoken_testv = []
    longUtter_testv = []
    
    testv_section_type_gpt = []
    testv_section_type = []
    testv_section_type_tar = []
    testv_precede = []
    
    testv_meeting_now = ""
    testv_first = []
    
    testv_hearing_phrase = []
    testv_comment_phrase = []
    testv_next_phrase = []
    testv_name_phrase = []
    
    for ind, example in enumerate(testv_list):
        spoken_testv.append(make_spoken(example))
        testv_commenttype += make_commenttype(example)
        testv_commenttype_tar += make_commenttype_tar(example)
        testv_llm_commenttype.append(make_plm_pred(example, ind, plm_pred["pred"]))
        
        testv_section_type_gpt.extend(make_section_type_gpt(example))
        testv_section_type.extend(make_section_type(example))
        testv_section_type_tar.extend(make_section_type_tar(example))
        
        testv_speaker_type_dict = _make_speakertype(example, testv_speaker_type_dict)
        longU, testv_speaker_long_dict = make_longUtter(example, testv_speaker_long_dict)
        longUtter_testv.append(longU)
        
        precede = make_precede(ind, testv_list)
        if precede:
            testv_precede.append(precede)
        first, testv_meeting_now = make_first(testv_meeting_now, example)
        if first:
            testv_first.append(first)
           
        testv_speaker_count_dict = get_speak_count(example, testv_speaker_count_dict)
        
        testv_hearing_phrase.append(make_phrase(example, hearing_signal))
        testv_comment_phrase.append(make_phrase(example, comment_signal))
        testv_next_phrase.append(make_phrase(example, next_signal))
        testv_name_phrase.append(make_phrase(example, name_signal))
    
    testv_speaker_long = make_long_utter_rate(testv_speaker_long_dict)
    
    testv_speaker_type = make_speakertype(testv_speaker_type_dict)
    for i in range(0, len(testv_speaker_type), 2):
        testv_speaker_type_tar += make_speaker_type_tar(testv_speaker_type[i])
    
    testv_speaker_low_count = make_low_count(testv_speaker_count_dict)
    testv_speaker_high_count = make_high_count(testv_speaker_count_dict)
    write_a_file(spoken_train, output + "/train/spoken.txt")
    write_a_file(longUtter_train, output + "/train/longUtter.txt")
    write_a_file(train_speaker_long, output + "/train/speaker_long.txt")
    write_a_file(contains_train, output + "/train/contains.txt")
    write_a_file(train_speaker_type, output + "/train/speaker_type_truth.txt")
    write_a_file(train_commenttype, output + "/train/commenttype_truth.txt")
    write_a_file(train_llm_commenttype, output + "/train/commenttype_llm.txt")
    write_a_file(train_speaker_type_tar, output + "/train/speaker_type_target.txt")
    write_a_file(train_commenttype_tar,output + "/train/commenttype_target.txt")
    
    write_a_file(train_section_type_gpt,output + "/train/sectiontype_obs.txt")
    write_a_file(train_section_type,output + "/train/sectiontype_truth.txt")
    write_a_file(train_section_type_tar, output + "/train/sectiontype_target.txt")
    
    write_a_file(train_precede,output + "/train/precedes.txt")
    write_a_file(train_first, output + "/train/first.txt")
    
    write_a_file(train_speaker_low_count, output + "/train/speaker_count_low.txt")
    write_a_file(train_speaker_high_count, output + "/train/speaker_count_high.txt")
    
    write_a_file(train_hearing_phrase, output + "/train/hearing_phrase.txt")
    write_a_file(train_comment_phrase, output + "/train/comment_phrase.txt")
    write_a_file(train_next_phrase, output + "/train/next_phrase.txt")
    write_a_file(train_name_phrase, output + "/train/name_phrase.txt")
    
    with open(output + "/train/vocab.json", "w") as f:
        json.dump(vocab, f)
    #
    write_a_file(spoken_test,output + "/eval/spoken.txt")
    write_a_file(longUtter_test,output + "/eval/longUtter.txt")
    write_a_file(test_speaker_long,output + "/eval/speaker_long.txt")
    write_a_file(contains_test,output + "/eval/contains.txt")
    write_a_file(test_speaker_type,output + "/eval/speaker_type_truth.txt")
    write_a_file(test_commenttype,output + "/eval/commenttype_truth.txt")
    write_a_file(test_llm_commenttype, output + "/eval/commenttype_llm.txt")
    write_a_file(test_speaker_type_tar,output + "/eval/speaker_type_target.txt")
    write_a_file(test_commenttype_tar,output + "/eval/commenttype_target.txt")
    
    write_a_file(test_section_type_gpt,output + "/eval/sectiontype_obs.txt")
    write_a_file(test_section_type,output + "/eval/sectiontype_truth.txt")
    write_a_file(test_section_type_tar,output + "/eval/sectiontype_target.txt")
    
    write_a_file(test_precede,output + "/eval/precedes.txt")
    write_a_file(test_first, output + "/eval/first.txt")
    
    write_a_file(test_speaker_low_count, output + "/eval/speaker_count_low.txt")
    write_a_file(test_speaker_high_count, output + "/eval/speaker_count_high.txt")
    
    write_a_file(test_hearing_phrase, output + "/eval/hearing_phrase.txt")
    write_a_file(test_comment_phrase, output + "/eval/comment_phrase.txt")
    write_a_file(test_next_phrase, output + "/eval/next_phrase.txt")
    write_a_file(test_name_phrase, output + "/eval/name_phrase.txt")
    #write_a_file(spoken_obs,"public_comments/0/eval/spoken.txt")

    write_a_file(spoken_testv,output + "/test/spoken.txt")
    write_a_file(longUtter_testv,output + "/test/longUtter.txt")
    write_a_file(testv_speaker_long,output + "/test/speaker_long.txt")
    write_a_file(contains_testv,output + "/test/contains.txt")
    write_a_file(testv_speaker_type,output + "/test/speaker_type_truth.txt")
    write_a_file(testv_commenttype,output + "/test/commenttype_truth.txt")
    write_a_file(testv_llm_commenttype, output + "/test/commenttype_llm.txt")
    write_a_file(testv_speaker_type_tar,output + "/test/speaker_type_target.txt")
    write_a_file(testv_commenttype_tar,output + "/test/commenttype_target.txt")
    
    write_a_file(testv_section_type_gpt,output + "/test/sectiontype_obs.txt")
    write_a_file(testv_section_type,output + "/test/sectiontype_truth.txt")
    write_a_file(testv_section_type_tar,output + "/test/sectiontype_target.txt")
    
    write_a_file(testv_precede,output + "/test/precedes.txt")
    write_a_file(testv_first, output + "/test/first.txt")
    
    write_a_file(testv_speaker_low_count, output + "/test/speaker_count_low.txt")
    write_a_file(testv_speaker_high_count, output + "/test/speaker_count_high.txt")
    
    write_a_file(testv_hearing_phrase, output + "/test/hearing_phrase.txt")
    write_a_file(testv_comment_phrase, output + "/test/comment_phrase.txt")
    write_a_file(testv_next_phrase, output + "/test/next_phrase.txt")
    write_a_file(testv_name_phrase, output + "/test/name_phrase.txt")

def LOO(data, output="../data/public_comments"):
    #random.shuffle(data)
    
    if not os.path.exists(output):
        os.makedirs(output)
    subdir_path = os.path.join(output, "mapping")
    if not os.path.exists(subdir_path):
        os.makedirs(subdir_path)
    with open(os.path.join(subdir_path, "all_id_mapping.json"), "w") as f:
        json.dump({"i2u": i2t, "u2i": t2i, "i2m": i2m, "m2i": m2i, "info2t": info2t, "t2info": t2info}, f)
    
    subdirs = ["train", "eval","test"]
    for subdir in subdirs:
        subdir_path = os.path.join(output, subdir)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)
    # print(len(data))
    training_set = data[:train_limit]
    test_set = data[train_limit: test_limit]
    testv_set = data[test_limit:]
    for t in test_set:
        print(len(t))
    print("===")
    for t in testv_set:
        print(len(t))
    print("===")
    for t in training_set:
        print(len(t))
    print("===")
    # All other groups are combined into the training set
    #training_set = [file for j in range(len(groups)) if j != i for file in groups[j]]
    training_set = sum(training_set, [])
    test_set = sum(test_set, [])
    testv_set = sum(testv_set, [])
    run_through(training_set, test_set, testv_set, output=output)
        
data = [truth[i] for i in truth]

#cross_val(data, cv=5, output="/home/shared/starter_code_public_comments/data/test_data_gen/AA")
#print(f"{out_dir}/{city}")
LOO(data, output=f"{out_dir}/{city}")

def make_longUtter(example, speaker_long_dict, threshold=50):
    """
    处理长文本
    
    Args:
        example: 示例数据
        speaker_long_dict: 说话者长文本字典
        threshold: 长文本阈值
        
    Returns:
        Tuple[str, Dict]: 长文本标记和更新后的字典
    """
    speaker = example["speaker"]
    M = example["meeting"]
    if len(example["text"].split()) > threshold:
        example["is_long_utter"] = 1
    else:
        example["is_long_utter"] = 0
        
    if speaker not in speaker_long_dict:
        speaker_long_dict[speaker] = {"long": 0, "total": 0, "meeting": M}
    speaker_long_dict[speaker]["long"] += example["is_long_utter"]
    speaker_long_dict[speaker]["total"] += 1
    
    utt_index = example["index"]
    is_long_utter = "1.0" if example["is_long_utter"] else "0.0"
    return get_line([M, utt_index, is_long_utter]), speaker_long_dict

def make_spoken(example):
    """
    处理说话者信息
    
    Args:
        example: 示例数据
        
    Returns:
        str: 说话者信息行
    """
    speaker = example["speaker"]
    M = example["meeting"]
    utt_index = example["index"]
    return get_line([M, utt_index, speaker, "1.0"])

def _make_speakertype(example, speaker_type_dict):
    """
    处理说话者类型
    
    Args:
        example: 示例数据
        speaker_type_dict: 说话者类型字典
        
    Returns:
        Dict: 更新后的说话者类型字典
    """
    speaker = example["speaker"]
    M = example["meeting"]
    try:
        if "Gov" in example["Speaker Role"]:
            role = "government"
        elif "Comm" in example["Speaker Role"]:
            role = "public"
        else:
            role = "other"
    except:
        role = "other"
    if speaker not in speaker_type_dict:
        speaker_type_dict[speaker] = {"government": 0, "public": 0, "other": 0, "meeting": M}
    speaker_type_dict[speaker][role] += 1
    return speaker_type_dict

def make_commenttype(example):
    """
    处理评论类型
    
    Args:
        example: 示例数据
        
    Returns:
        List[str]: 评论类型行列表
    """
    M = example["meeting"]
    utt_index = example["index"]
    return_list = []
    if example["is_public_comment"]:
        return_list.append(get_line([M, str(utt_index), "PC", "1.0"]))
        return_list.append(get_line([M, str(utt_index), "PH", "0.0"]))
        return_list.append(get_line([M, str(utt_index), "Other", "0.0"]))
    else:
        return_list.append(get_line([M, str(utt_index), "PC", "0.0"]))
        if example["is_public_hearing"]:
            return_list.append(get_line([M, str(utt_index), "PH", "1.0"]))
            return_list.append(get_line([M, str(utt_index), "Other", "0.0"]))
        else:
            return_list.append(get_line([M, str(utt_index), "PH", "0.0"]))
            return_list.append(get_line([M, str(utt_index), "Other", "1.0"]))
    
    return return_list

def make_speakertype(speaker_type_dict):
    """
    生成说话者类型
    
    Args:
        speaker_type_dict: 说话者类型字典
        
    Returns:
        List[str]: 说话者类型行列表
    """
    speaker_type = []
    
    for speaker in speaker_type_dict:
        M = speaker_type_dict[speaker]["meeting"]
        if speaker_type_dict[speaker]["government"] >= speaker_type_dict[speaker]["public"] and \
        speaker_type_dict[speaker]["government"] >= speaker_type_dict[speaker]["other"]:
            st = "government"
        elif speaker_type_dict[speaker]["public"] >= speaker_type_dict[speaker]["government"] and \
        speaker_type_dict[speaker]["public"] >= speaker_type_dict[speaker]["other"]:
            st = "public"
        else:
            st = "other"
        if st == "public":
            speaker_type.append(get_line([M, speaker, "public", "1.0"]))
            speaker_type.append(get_line([M, speaker, "other", "0.0"]))
        else:
            speaker_type.append(get_line([M, speaker, "public", "0.0"]))
            speaker_type.append(get_line([M, speaker, "other", "1.0"]))
    return speaker_type    

def make_long_utter_rate(speaker_long_dict):
    """
    生成长文本比率
    
    Args:
        speaker_long_dict: 说话者长文本字典
        
    Returns:
        List[str]: 长文本比率行列表
    """
    speaker_long = []
    for speaker in speaker_long_dict:
        speaker_long.append(get_line([speaker_long_dict[speaker]["meeting"], speaker, 
                                    str(speaker_long_dict[speaker]["long"] / speaker_long_dict[speaker]["total"])]))
    return speaker_long

def make_contain(train_list, val_list, test_list):
    """
    生成包含关系
    
    Args:
        train_list: 训练数据列表
        val_list: 验证数据列表
        test_list: 测试数据列表
        
    Returns:
        Tuple[List[str], List[str], List[str], List[str]]: 包含关系行列表和词汇表
    """
    example_list = train_list + val_list + test_list
    contains_train, contains_val, contains_test = [], [], []
    docs = [example["text"] for example in example_list]
    utt_index = [example["index"] for example in example_list]
    M = [example["meeting"] for example in example_list]
    cvv = CustomVectorizer()
    new_docs = cvv.prepare_doc(docs)
    matrics = cvv.fit_transform(new_docs).toarray()
    vocab = cvv.get_feature_names_out()
    non_zero_indices = np.argwhere(matrics != 0)
    words = vocab[non_zero_indices[:, 1]]
    temp = np.cumsum(np.count_nonzero(matrics, axis=1))[:-1]
    matched_words = np.split(words, temp)
    for i, entry in tqdm(enumerate(matched_words)):
        for j in entry:
            if i < len(train_list):
                contains_train.append(get_line([M[i], str(utt_index[i]), j, "1.0"]))
            elif i < len(train_list) + len(val_list):
                contains_val.append(get_line([M[i], str(utt_index[i]), j, "1.0"]))
            else:
                contains_test.append(get_line([M[i], str(utt_index[i]), j, "1.0"]))

    return contains_train, contains_val, contains_test, vocab.tolist()

def get_speak_count(example, speaker_count_dict):
    """
    获取说话者计数
    
    Args:
        example: 示例数据
        speaker_count_dict: 说话者计数字典
        
    Returns:
        Dict: 更新后的说话者计数字典
    """
    M = example["meeting"]
    speaker = example["speaker"]
    if M + "_" + speaker not in speaker_count_dict:
        speaker_count_dict[M + "," + speaker] = 0
    speaker_count_dict[M + "," + speaker] += 1
    return speaker_count_dict
    
def make_low_count(speaker_count_dict):
    """
    生成低计数标记
    
    Args:
        speaker_count_dict: 说话者计数字典
        
    Returns:
        List[str]: 低计数标记行列表
    """
    lines = []
    for k,v in speaker_count_dict.items():
        meeting_id = k.split(',')[0]
        speaker_id = k.split(',')[1]
        if 2 > v:  
            lines.append(get_line([meeting_id, speaker_id, '1.0']))
        else:
            lines.append(get_line([meeting_id, speaker_id, '0.0']))
    return lines

def make_high_count(speaker_count_dict):
    """
    生成高计数标记
    
    Args:
        speaker_count_dict: 说话者计数字典
        
    Returns:
        List[str]: 高计数标记行列表
    """
    lines = []
    for k,v in speaker_count_dict.items():
        meeting_id = k.split(',')[0]
        speaker_id = k.split(',')[1]
        if 2 > v:  
            lines.append(get_line([meeting_id, speaker_id, '1.0']))
        else:
            lines.append(get_line([meeting_id, speaker_id, '0.0']))
    return lines

def make_section_type_gpt(example):
    """
    生成GPT部分类型
    
    Args:
        example: 示例数据
        
    Returns:
        List[str]: 部分类型行列表
    """
    M = example["meeting"]
    utt_index = example["index"]
    mapping = {
            "other": "Other",
            "public_comment": "PC",
            "public_hearing": "PH",
    }
    assigning = {
            "PC": "0.0",
            "Other": "0.0",
            "PH": "0.0",
    }
    section_type = mapping[example["section_type_gpt"]] 
    assigning[section_type] = "1.0"
    return [get_line([M, str(utt_index), "PC", assigning["PC"]]),
            get_line([M, str(utt_index), "Other", assigning["Other"]]),
            get_line([M, str(utt_index), "PH", assigning["PH"]])
           ]

def make_section_type(example):
    """
    生成部分类型
    
    Args:
        example: 示例数据
        
    Returns:
        List[str]: 部分类型行列表
    """
    M = example["meeting"]
    utt_index = example["index"]
    mapping = {
            "Public Comment": "PC",
            "Both": "PC", 
            "Other": "Other",
            "Public Hearing": "PH",
            "Dummy": "Other"
    }
    assigning = {
            "PC": "0.0",
            "Other": "0.0",
            "PH": "0.0",
    }
    section_type = mapping[example["Meeting Section"]]
    assigning[section_type] = "1.0"
    return [get_line([M, str(utt_index), "PC", assigning["PC"]]),
            get_line([M, str(utt_index), "Other", assigning["Other"]]),
            get_line([M, str(utt_index), "PH", assigning["PH"]])
           ]

def make_plm_pred(example, ind, plm_pred):
    """
    生成PLM预测
    
    Args:
        example: 示例数据
        ind: 索引
        plm_pred: PLM预测结果
        
    Returns:
        str: PLM预测行
    """
    M = example["meeting"]
    utt_index = example["index"]
    mapping = ["Other", "PC", "PH"] 
    comment_type = mapping[plm_pred[ind]] 
    return get_line([M, str(utt_index), comment_type, "1.0"])

def make_section_type_tar(example):
    """
    生成部分类型目标
    
    Args:
        example: 示例数据
        
    Returns:
        List[str]: 部分类型目标行列表
    """
    M = example["meeting"]
    utt_index = example["index"]
    return [get_line([M, str(utt_index), "PC"]), 
            get_line([M, str(utt_index), "PH"]), 
            get_line([M, str(utt_index), "Other"])]

def make_commenttype_tar(example):
    """
    生成评论类型目标
    
    Args:
        example: 示例数据
        
    Returns:
        List[str]: 评论类型目标行列表
    """
    M = example["meeting"]
    utt_index = example["index"]
    return [get_line([M, str(utt_index), "PC"]), 
            get_line([M, str(utt_index), "PH"]), 
            get_line([M, str(utt_index), "Other"])]

def make_speaker_type_tar(s):
    """
    生成说话者类型目标
    
    Args:
        s: 说话者信息字符串
        
    Returns:
        List[str]: 说话者类型目标行列表
    """
    M = s.split('\t')[0]
    speaker = s.split('\t')[1]
    return [get_line([M, speaker, "public"]), 
            get_line([M, speaker, "other"])]

def make_precede(i, example_list):
    """
    生成前序关系
    
    Args:
        i: 索引
        example_list: 示例列表
        
    Returns:
        Optional[str]: 前序关系行
    """
    M = example_list[i]["meeting"]
    utt_index = example_list[i]["index"]
    if i + 1 < len(example_list):
        next_M = example_list[i+1]["meeting"]
        next_utt_index = example_list[i+1]["index"]
        if next_M == M:
            return get_line([M, utt_index, next_utt_index, "1.0"])
    return None
    
def make_first(now_meeting, example):
    """
    生成首次标记
    
    Args:
        now_meeting: 当前会议
        example: 示例数据
        
    Returns:
        Tuple[Optional[str], str]: 首次标记行和更新后的会议
    """
    M = example["meeting"]
    utt_index = example["index"]
    if now_meeting != M:
        now_meeting = M
        return get_line([M, utt_index]), now_meeting
    return None, now_meeting
    
def make_phrase(example, phrases):
    """
    生成短语标记
    
    Args:
        example: 示例数据
        phrases: 短语列表
        
    Returns:
        str: 短语标记行
    """
    M = example["meeting"]
    utt_index = example["index"]
    txt = example["text"]
    val = "0.0"
    for p in phrases:
        if p in txt.lower():
            val = "1.0"
    return get_line([M, utt_index, val])