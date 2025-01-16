import whisperx
import gc
import json
import torch

def use_whisperx(device, audio_file, hf_token, batch_size=16, compute_type="float16", model_name="large-v3"):

    # 1. Transcribe with original whisper (batched)

    if ":" in device:
        dev, dev_ind = device.split(":")[0], device.split(":")[1]
        print("here")
        model = whisperx.load_model(model_name, dev, device_index=int(dev_ind), compute_type=compute_type, asr_options={"initial_prompt": "Add Punctuation:"})
        # model = whisperx.load_model(model_name, dev, device_index=int(dev_ind), compute_type=compute_type, asr_options={"initial_prompt": "Add Punctuation:"})
    else:
        print("there")
        # model = whisperx.load_model(model_name, device, device_index=[1,2,3,4], compute_type=compute_type, asr_options={"initial_prompt": "Hello."})
        model = whisperx.load_model(model_name, device, compute_type=compute_type, asr_options={"initial_prompt": "Hello."})

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size, language="en")

    # delete model if low on GPU resources
    gc.collect()
    torch.cuda.empty_cache()
    del model

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # delete model if low on GPU resources
    gc.collect()
    torch.cuda.empty_cache()
    del model_a

    # 3. Assign speaker labels
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)

    # add min/max number of speakers if known
    diarize_segments = diarize_model(audio_file)

    result = whisperx.assign_word_speakers(diarize_segments, result)
    
    return result