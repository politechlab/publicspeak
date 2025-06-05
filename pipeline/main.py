import os
import argparse
from typing import Dict, Any, Optional
from pathlib import Path
from collections import defaultdict
from pipeline.transcribe.transcribe import use_whisperx, save_transcription_result
from pipeline.llm_processor.llm_processor import process_with_llm
from pipeline.public_speech_extractor.extractor import clean_and_find_manager, extract_public, save_extraction_result
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,7"

def parse_args():
    parser = argparse.ArgumentParser(description='Public Speech Processing Pipeline')
    
    # 转写相关参数
    parser.add_argument('--mode', type=str, required=True, choices=['transcribe', 'process', 'extract', 'full'],
                      help='Pipeline mode')
    parser.add_argument('--audio_file', type=str, help='Input audio file path')
    parser.add_argument('--ts_path', type=str, help='Transcription file path')
    parser.add_argument('--model_name', type=str, default='large-v2',
                      help='WhisperX model name')
    parser.add_argument('--device', type=str, default='cuda:1',
                      help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size for transcription')
    parser.add_argument('--compute_type', type=str, default='float16',
                      help='Compute type for WhisperX')
    
    # LLM处理相关参数
    parser.add_argument('--gpt_version', type=str, default='gpt-4',
                      help='GPT model version')
    parser.add_argument('--cut_off_th', type=int, default=50,
                      help='Threshold for cut_off function')
    
    # 输出相关参数
    parser.add_argument('--output_dir', type=str, default='output',
                      help='Output directory')
    
    return parser.parse_args()

def get_transcription_result(args) -> Dict[str, Any]:
    """
    Get transcription result
    
    Args:
        args: Command line arguments
        
    Returns:
        Dict[str, Any]: Transcription result
    """
    model_name = args.model_name
    device = args.device
    hf_token = os.getenv("HF_TOKEN")
    audio_file = args.audio_file
    batch_size = args.batch_size
    compute_type = args.compute_type
    
    try:
        return use_whisperx(device, audio_file, hf_token, batch_size, compute_type, model_name)
    except KeyError:
        print(f"No word is unknown.")
        return {}

def save_transcription_result(result: Dict[str, Any], output_dir: str, audio_file: str) -> str:
    """
    Save transcription result to file
    
    Args:
        result: Transcription result
        output_dir: Output directory
        audio_file: Input audio file path
        
    Returns:
        str: Path to the saved file
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    json_name = os.path.splitext(audio_file)[0].split("/")[-1] + ".json"
    output_path = os.path.join(output_dir, json_name)
    
    with open(output_path, 'w') as f:
        json.dump(result, f)
    
    return output_path

def get_llm_result(args, ts_path: str) -> Dict[str, Any]:
    """
    Get LLM processing result
    
    Args:
        args: Command line arguments
        ts_path: Path to the transcription file
        
    Returns:
        Dict[str, Any]: LLM processing result
    """
    manager, merged_data = clean_and_find_manager(ts_path)
    cnt = 0
    idx_mapping = {}
    content = ""
    result = {}
    result["merged_data"] = merged_data
    
    # Prepare content with numbering
    for i, item in enumerate(merged_data):
        content += str(cnt) + ". " + item["text"].strip() + "\n"
        idx_mapping[str(cnt) + ". "] = i
        cnt += 1
    
    result["idx_map"] = idx_mapping
    
    # Process with LLM
    result["triggers"] = process_with_llm(
        content=content,
        model=args.gpt_version,
        args=args
    )
    
    return result

def save_llm_result(result: Dict[str, Any], ts_path: str, mode: str) -> str:
    """
    Save LLM processing result to file
    
    Args:
        result: LLM processing result
        ts_path: Path to the transcription file
        mode: Processing mode
        
    Returns:
        str: Path to the saved file
    """
    trigger_path = os.path.splitext(ts_path)[0] + f"_trigger{'_general' if mode != 'find_public_trigger' else ''}.json"
    with open(trigger_path, "w") as f:
        json.dump(result, f)
    
    return trigger_path

def get_extraction_result(args, trigger_path: str) -> Dict[str, Any]:
    """
    Get public speech extraction result
    
    Args:
        args: Command line arguments
        trigger_path: Path to the trigger file
        
    Returns:
        Dict[str, Any]: Extraction result
    """
    with open(trigger_path, "r") as f:
        triggers = json.load(f)
    
    this_merged = triggers["merged_data"]
    this_mapping = triggers["idx_map"]
    this_triggers = triggers["triggers"]
    
    return extract_public(args, this_merged, this_mapping, this_triggers)

def save_extraction_result(result: Dict[str, Any], trigger_path: str) -> str:
    """
    Save public speech extraction result to file
    
    Args:
        result: Extraction result
        trigger_path: Path to the trigger file
        
    Returns:
        str: Path to the saved file
    """
    public_path = os.path.splitext(trigger_path)[0].replace("_trigger", "_public") + ".json"
    with open(public_path, "w") as f:
        json.dump(result, f)
    
    return public_path

def run_transcribe(args) -> str:
    """Run transcription process"""
    result = get_transcription_result(args)
    return save_transcription_result(result, args.output_dir, args.audio_file)

def run_llm_processing(args, ts_path: str) -> str:
    """Run LLM processing"""
    result = get_llm_result(args, ts_path)
    return save_llm_result(result, ts_path, args.mode)

def run_extraction(args, trigger_path: str) -> str:
    """Run public speech extraction"""
    result = get_extraction_result(args, trigger_path)
    return save_extraction_result(result, trigger_path)

def main():
    """Main function to run the pipeline"""
    args = parse_args()
    
    if args.mode == "transcribe":
        run_transcribe(args)
    
    elif args.mode in ["find_public_trigger", "find_public_trigger_general"]:
        if not args.ts_path:
            raise ValueError("ts_path is required for find_public_trigger modes")
        trigger_path = run_llm_processing(args, args.ts_path)
        run_extraction(args, trigger_path)
    
    elif args.mode == "full":
        if not args.audio_file:
            raise ValueError("audio_file is required for full mode")
        ts_path = run_transcribe(args)
        trigger_path = run_llm_processing(args, ts_path)
        run_extraction(args, trigger_path)

if __name__ == "__main__":
    main() 