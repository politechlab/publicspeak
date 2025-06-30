# PublicSpeak Pipeline

This pipeline provides a comprehensive data processing workflow for the PublicSpeak project, including audio transcription, LLM processing, public speech extraction, PLM training, and data generation.

## 📁 Project Structure

```
publicspeak/pipeline/
├── __init__.py                    # Package initialization
├── main.py                        # Main pipeline controller
├── transcribe/                    # Audio transcription module
│   ├── __init__.py
│   └── transcribe.py              # WhisperX transcription
├── PLM/                          # Pre-trained Language Model module
│   └── finetuned_one_val_out.py  # PLM training and inference
├── llm_processor/                 # LLM processing module
│   ├── __init__.py
│   ├── config.py                  # LLM configuration
│   ├── llm_processor.py           # Main LLM processor
│   └── llm_utils.py               # LLM utilities
├── public_speech_extractor/       # Public speech extraction module
│   ├── __init__.py
│   └── extractor.py               # Speech extraction logic
├── generate_processed_data/       # Data generation module
│   └── generate_processed_data.py # Data processing and generation
├── plm_processor/                 # PLM processing utilities
└── process_data_from_cdp/        # CDP data processing
```

## 🚀 Quick Start

One can always find configurations in the folder config, or one can input using command line options.

### 1. Audio Transcription

```bash
# Transcribe audio file
python -m pipeline.main --mode transcribe \
    --audio_file path/to/audio.wav \
    --output_dir output
```

### 2. LLM Processing

```bash
# Process transcription with LLM
python -m pipeline.main --mode find_public_trigger \
    --ts_path path/to/transcription.json \
    --output_dir output
```

### 3. PLM Training

```bash
# Train PLM model
python -m pipeline.main --mode plm \
    --lr 2e-5 \
    --epoch 10 \
    --seed 42 \
```

### 4. PLM Prediction

```bash
# Run PLM prediction on transcript
python -m pipeline.main --mode plm_predict \
    --ts_path path/to/transcription.json \
```

### 5. Data Generation

```bash
# Generate processed data for PSL
python -m pipeline.main --mode generate_data \
    --data_mode full \
    --train_file train_data.json \
    --test_file test_data.json \
    --eval_file eval_data.json \
    --output_dir output
```

### 6. Full Pipeline

```bash
# Run complete pipeline (transcribe + LLM + extract + PLM)
python -m pipeline.main --mode full \
    --audio_file path/to/audio.wav \
    --output_dir output
```

## ⚙️ Parameters

### Transcription Parameters
- `--audio_file`: Input audio file path
- `--model_name`: WhisperX model name (default: large-v2)
- `--device`: Device to use (default: cuda:1)
- `--batch_size`: Batch size for transcription
- `--compute_type`: Compute type for WhisperX (default: float16)

### LLM Parameters
- `--gpt_version`: GPT model version (default: gpt-4)
- `--cut_off_th`: Threshold for cut_off function
- `--long_text_th`: Threshold for long text detection
- `--ratio_count`: Threshold for long utterance ratio

### PLM Parameters
- `--plm_model_name`: PLM model name (default: roberta-base)
- `--lr`: Learning rate (default: 2e-5)
- `--epoch`: Number of epochs (default: 10)
- `--seed`: Random seed (default: 42)
- `--plm_batch_size`: PLM batch size
- `--save_plm_model`: Whether to save PLM model

### Data Generation Parameters
- `--data_mode`: Processing mode (full/test_only)
- `--train_file`: Training data file name
- `--test_file`: Test data file name
- `--eval_file`: Evaluation data file name
- `--plm_file_name`: PLM prediction file name

### General Parameters
- `--output_dir`: Output directory (default: output)
- `--ts_path`: Transcription file path
- `--psl_data_dir`: PSL data directory


## 🔧 Configuration

The pipeline uses configuration files in the `config` directory:

- `config/settings.py`: General settings and model configurations
- `config/paths.py`: File and directory paths

Key configuration options:
- `Settings.WHISPER_MODEL_NAME`: Default WhisperX model
- `Settings.GPT_VERSION`: Default GPT model version
- `Settings.MODEL_NAME`: Default PLM model name
- `Paths.RAW_TRAIN_DIR`: Raw training data directory
- `Paths.RAW_TEST_DIR`: Raw test data directory

## 📦 Dependencies

- WhisperX for audio transcription
- OpenAI GPT models for LLM processing
- Transformers for PLM models
- PyTorch for deep learning
- NumPy and pandas for data processing
- Other dependencies as specified in requirements

## ⚠️ Important Notes

1. **Audio Files**: Supported formats include WAV, MP3, M4A
2. **GPU Requirements**: PLM training and inference require CUDA-compatible GPU
3. **API Keys**: LLM processing requires OpenAI API key (set as environment variable)
4. **Model Files**: Ensure PLM models are downloaded before training
5. **Data Format**: Input data should follow the specified JSON format
6. **Memory**: Large audio files may require significant memory

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use smaller models
2. **API Rate Limits**: Implement retry logic for LLM calls
3. **File Not Found**: Check file paths and ensure files exist
4. **Model Download Issues**: Verify internet connection and model names

### Debug Mode

Add verbose logging to debug issues:
```bash
python -m pipeline.main --mode transcribe \
    --audio_file audio.wav \
    --output_dir output \
    --verbose
```

## 📄 Output Files

The pipeline generates various output files:

- `*_transcription.json`: WhisperX transcription results
- `*_trigger.json`: LLM processing results
- `*_extraction.json`: Public speech extraction results
- `*_plm.json`: PLM prediction results
- `processed_data/`: Generated data for PSL models

## 📄 License

This project follows the license terms specified in the project root directory. 