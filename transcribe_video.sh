VIDEO_NAME="Ann Arbor City Council 112023.mp4"
VIDEO_PATH="/home/shared/turbo_data/localgov/videos/2023/03000/${VIDEO_NAME}"
WAV_DIR="../../data/audio/"
TRANSCRIPTS_DIR="../../data/transcripts/"
WAV_NAME="${VIDEO_NAME%.*}.wav"  # 更通用的文件扩展名替换方式

# 执行转换命令
python main.py --mode to_wav --wav_input "${VIDEO_PATH}" --wav_output "${WAV_DIR}" && \
python main.py --mode transcribe --device cuda --audio_file "${WAV_DIR}${WAV_NAME}" --ts_output_folder "${TRANSCRIPTS_DIR}"
