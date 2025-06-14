import os
import argparse
from typing import Dict, Any, Optional
from pathlib import Path
from collections import defaultdict
from pipeline.transcribe.transcribe import use_whisperx, save_transcription_result
from pipeline.llm_processor.llm_processor import process_with_llm
from pipeline.public_speech_extractor.extractor import clean_and_find_manager, extract_public, save_extraction_result
import json
from config import Paths, Settings


def parse_args():
    parser = argparse.ArgumentParser(description='Public Speech Processing Pipeline')
    
    # 转写相关参数
    parser.add_argument('--mode', type=str, required=True, 
                      choices=['transcribe', 'process', 'extract', 'plm', 'full', 'generate_data'],
                      help='Pipeline mode')
    parser.add_argument('--audio_file', type=str, help='Input audio file path')
    parser.add_argument('--ts_path', type=str, help='Transcription file path')
    parser.add_argument('--model_name', type=str, default=Settings.WHISPER_MODEL_NAME,
                      help='WhisperX model name')
    parser.add_argument('--device', type=str, default='cuda:1',
                      help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=Settings.BATCH_SIZE,
                      help='Batch size for transcription')
    parser.add_argument('--compute_type', type=str, default='float16',
                      help='Compute type for WhisperX')
    
    # LLM处理相关参数
    parser.add_argument('--gpt_version', type=str, default='gpt-4',
                      help='GPT model version')
    parser.add_argument('--cut_off_th', type=int, default=50,
                      help='Threshold for cut_off function')
    parser.add_argument('--long_text_th', type=int, default=50,
                      help='Threshold for count_long_text_ratio function')
    parser.add_argument('--ratio_count', type=float, default=0.5,
                       help='Threshold for identifying long utterance ratio')
    
    # PLM相关参数
    parser.add_argument('--plm_model_name', type=str, default=Settings.MODEL_NAME,
                      help='PLM model name')
    parser.add_argument('--city', type=str, default=Settings.CITY,
                      help='City for PLM training')
    parser.add_argument('--lr', type=float, default=Settings.LEARNING_RATE,
                      help='Learning rate for PLM')
    parser.add_argument('--epoch', type=int, default=Settings.EPOCHS,
                      help='Number of epochs for PLM')
    parser.add_argument('--seed', type=int, default=Settings.SEED,
                      help='Random seed for PLM')
    parser.add_argument('--plm_batch_size', type=int, default=Settings.PLM_BATCH_SIZE,
                      help='Batch size for transcription')
    
    # 生成数据相关参数 TODO
    parser.add_argument('--plm_file_name', type=str, default="AA_pred_LOO_roberta.json",
                      help='PLM prediction file name')
    
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
    # device = args.device
    device = "cuda"
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

def get_plm_result(args, ts_path: str) -> Dict[str, Any]:
    """
    Get PLM processing result
    
    Args:
        args: Command line arguments
        ts_path: Path to the transcription file
        
    Returns:
        Dict[str, Any]: PLM processing result
    """
    # 读取转录数据
    with open(ts_path) as f:
        transcript_data = json.load(f)
    
    # 使用PLM处理转录数据
    from pipeline.PLM.finetuned_one_val_out import process_transcript
    result = process_transcript(
        transcript_data=transcript_data,
        model_name=args.plm_model_name,
        device=args.device
    )
    
    return result

def save_plm_result(result: Dict[str, Any], ts_path: str) -> str:
    """
    Save PLM processing result to file
    
    Args:
        result: PLM processing result
        ts_path: Path to the transcription file
        
    Returns:
        str: Path to the saved file
    """
    plm_path = os.path.splitext(ts_path)[0] + "_plm.json"
    with open(plm_path, "w") as f:
        json.dump(result, f, indent=2)
    
    return plm_path

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

def run_plm_processing(args, ts_path: str) -> str:
    """Run PLM processing"""
    from pipeline.PLM.finetuned_one_val_out import main as plm_main
    plm_main(args)  # 直接运行PLM的main函数
    return ts_path  # 返回原始ts_path，因为PLM的结果会保存在配置的路径中

def run_generate_data(args) -> None:
    """Run data generation process"""
    from pipeline.generate_processed_data.generate_processed_data import generate_processed_data
    
    # 设置数据目录
    data_dir = Paths.DATA_DIR
    
    # 调用generate_processed_data函数
    generate_processed_data(
        city=args.city,
        data_dir=data_dir,
        output_dir=os.path.join(args.output_dir, args.city),
        plm_file_name=args.plm_file_name,
        seed=args.seed
    )

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
    
    elif args.mode == "plm":
        run_plm_processing(args, None)  # ts_path在这里不需要
    
    elif args.mode == "generate_data":
        run_generate_data(args)
    
    elif args.mode == "full":
        if not args.audio_file:
            raise ValueError("audio_file is required for full mode")
        ts_path = run_transcribe(args)
        trigger_path = run_llm_processing(args, ts_path)
        run_extraction(args, trigger_path)
        run_plm_processing(args, None)  # ts_path在这里不需要

if __name__ == "__main__":
    main() 