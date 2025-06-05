from typing import Dict, Any, List, Tuple
import json
from collections import defaultdict
from ..llm_processor.llm_processor import process_with_llm
from ..llm_processor.config import HALLUCINATION_INDICATORS

def clean_and_find_manager(ts_path: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Clean transcription text and find the manager
    
    Args:
        ts_path: transcription file path
    
    Returns:
        Tuple[str, List[Dict[str, Any]]]: (manager ID, cleaned data)
    """
    with open(ts_path, 'r') as f:
        transcript_data = json.load(f)
    
    # 清理文本
    clean_data = []
    for utterance in transcript_data['segments']:
        utterance["text"] = str(utterance["text"])
        if not any(indicator in utterance["text"].lower() for indicator in HALLUCINATION_INDICATORS) and len(utterance["text"]) > 0:
            if "speaker" not in utterance:
                utterance["speaker"] = "UNKNOWN"
            clean_data.append(utterance)
    
    # 合并相邻话语
    merged_data = []
    current_speaker = None
    current_text = ""
    current_end = 0
    temp_start = 0
    
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
            merged_data.append({
                "start": temp_start,
                "end": current_end,
                "speaker": current_speaker,
                "text": current_text
            })
            current_speaker = entry['speaker']
            current_text = entry['text']
            temp_start = entry['start']
            current_end = entry['end']
    
    # 添加最后一个条目
    if current_text:
        merged_data.append({
            "start": temp_start,
            "end": clean_data[-1]["end"],
            "speaker": current_speaker,
            "text": current_text
        })
    
    # 统计每个说话人的话语数量
    merged_utterance_counts = defaultdict(int)
    for entry in merged_data:
        merged_utterance_counts[entry['speaker']] += 1
    
    # 找出说话最多的前三个说话人
    top_3_keys = [k for k, v in sorted(merged_utterance_counts.items(), key=lambda item: item[1], reverse=True)[:3]]
    
    # 通过关键词识别主持人
    for item in clean_data:
        if item['speaker'] in top_3_keys:
            if "Pledge of Allegiance" in item['text'] or "roll call" in item['text']:
                return item['speaker'], merged_data
    
    # 如果没有找到关键词，返回说话最多的说话人
    return top_3_keys[0], merged_data

def extract_public(args: Any, merged_data: List[Dict[str, Any]], idx_mapping: Dict[str, int], triggers: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract public speech content
    
    Args:
        args: arguments object
        merged_data: merged data
        idx_mapping: index mapping
        triggers: triggers
    
    Returns:
        Dict[str, Any]: extraction result
    """
    result = {
        'public_speeches': [],
        'metadata': {
            'total_segments': len(merged_data),
            'triggers_used': list(triggers.keys())
        }
    }
    
    # Extract public speech segments based on triggers
    for trigger, indices in triggers.items():
        for idx in indices:
            if str(idx) in idx_mapping:
                segment_idx = idx_mapping[str(idx)]
                if segment_idx < len(merged_data):
                    result['public_speeches'].append({
                        'trigger': trigger,
                        'content': merged_data[segment_idx],
                        'index': segment_idx
                    })
    
    return result

def save_extraction_result(result: Dict[str, Any], output_path: str) -> None:
    """
    Save extraction result
    
    Args:
        result: extraction result
        output_path: output path
    """
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2) 