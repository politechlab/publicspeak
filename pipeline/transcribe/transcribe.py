import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,4,7"
import json
from typing import Dict, Any
import whisperx
import gc
import torch

print("CUDA Available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else "N/A")

def use_whisperx(device: str, audio_file: str, hf_token: str, batch_size: int, compute_type: str, model_name: str) -> Dict[str, Any]:
    """
    use WhisperX for transcription
    
    Args:
        device: device (cuda/cpu)
        audio_file: audio file path
        hf_token: HuggingFace token
        batch_size: batch size
        compute_type: compute type
        model_name: model name
    
    Returns:
        Dict[str, Any]: transcription result
    """
    # load model
    # TODO: Verify feasibility
    # model = whisperx.load_model(model_name, device, compute_type=compute_type)
    print(device)
    if ":" in device:
        dev, dev_ind = device.split(":")[0], device.split(":")[1]
        print("============")
        print(dev, dev_ind)
        model = whisperx.load_model(model_name, dev, device_index=int(dev_ind), compute_type=compute_type, asr_options={"initial_prompt": "Add Punctuation:"})
    else:
        model = whisperx.load_model(model_name, device, compute_type=compute_type, asr_options={"initial_prompt": "Hello."})
    
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size, language="en")

    # clean GPU resources
    gc.collect()
    torch.cuda.empty_cache()
    del model

    # align timestamps
    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # clean GPU resources
    gc.collect()
    torch.cuda.empty_cache()
    del model_a

    # speaker diarization
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
    diarize_segments = diarize_model(audio_file)
    
    # assign speakers to words
    result = whisperx.assign_word_speakers(diarize_segments, result)
    
    return result

def save_transcription_result(result: Dict[str, Any], output_path: str, filename: str) -> None:
    """
    save transcription result to JSON file
    
    Args:
        result: transcription result
        output_path: output directory
        filename: file name
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    json_name = os.path.splitext(filename)[0] + ".json"
    with open(os.path.join(output_path, json_name), 'w') as f:
        json.dump(result, f) 