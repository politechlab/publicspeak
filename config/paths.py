from pathlib import Path

class Paths:
    # base path
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    
    # raw data path
    RAW_DIR = DATA_DIR / "raw"
    RAW_TRAIN_DIR = DATA_DIR / "raw_train"
    RAW_EVAL_DIR = DATA_DIR / "raw_val"
    RAW_TEST_DIR = DATA_DIR / "raw_test"
    AUDIO_DIR = RAW_DIR / "audio"
    VIDEO_DIR = RAW_DIR / "video"
    TRANSCRIPTS_DIR = RAW_DIR / "transcripts"
    
    # video and audio processing paths
    VIDEO_POOL_DIR = DATA_DIR / "video_pool"
    WAV_INPUT_DIR = AUDIO_DIR / "input"
    WAV_OUTPUT_DIR = AUDIO_DIR / "output"
    TS_OUTPUT_DIR = TRANSCRIPTS_DIR / "new_ts_output"
    PSL_DATA_DIR = DATA_DIR / "processed_train_data_TEST"
    
    # processed data path
    PROCESSED_DIR = DATA_DIR / "processed"
    LLM_DIR = PROCESSED_DIR / "LLM_indicators"
    PLM_DIR = DATA_DIR / "PLM_indicators"
    PUBLIC_COMMENTS_DIR = PROCESSED_DIR / "public_comments"
    
    # generated data path
    GENERATED_DIR = DATA_DIR / "generated"
    TRAIN_DIR = GENERATED_DIR / "train"
    EVAL_DIR = GENERATED_DIR / "eval"
    TEST_DIR = GENERATED_DIR / "test"
    
    # PSL model paths
    PSL_MODEL_DIR = BASE_DIR / "model"
    PSL_TRAINING_DIR = PSL_MODEL_DIR / "training"
    PSL_INFERENCE_DIR = PSL_MODEL_DIR / "inference"
    PSL_PAPER_REPRODUCE_DIR = PSL_MODEL_DIR / "paper_reproduce"
    
    # PSL data directories
    PSL_PROCESSED_TEST_DATA = DATA_DIR / "processed_test_data" / "AA"
    PSL_GENERATED_TRAIN_DATA = DATA_DIR / "processed_train_data" / "AA" / "train"
    
    # PSL output directories
    PSL_LEARNT_WEIGHT_DIR = PSL_TRAINING_DIR / "learnt_weight"
    PSL_TEMP_LEARN_DIR = PSL_TRAINING_DIR / "temp_learn"
    PSL_TEMP_INFER_DIR = PSL_TRAINING_DIR / "temp_infer"
    PSL_OUTPUT_DIR = PSL_INFERENCE_DIR / "output"
    PSL_PAPER_REPRODUCE_OUTPUT = PSL_PAPER_REPRODUCE_DIR / "output"
    
    # PSL config files
    PSL_INIT_WEIGHT_FILE = PSL_TRAINING_DIR / "init_weight_file.json"
    PSL_WEIGHT_FILE = PSL_INFERENCE_DIR / "weight_file.json"
    
    @classmethod
    def get_psl_train_dir(cls):
        """Get PSL training data directory"""
        return cls.PSL_GENERATED_TRAIN_DATA
    
    @classmethod
    def get_psl_eval_dir(cls):
        """Get PSL evaluation data directory"""
        return cls.PSL_PROCESSED_TEST_DATA
    
    @classmethod
    def create_dirs(cls):
        """create all necessary directories"""
        for path in [
            cls.RAW_DIR, cls.AUDIO_DIR, cls.VIDEO_DIR, cls.TRANSCRIPTS_DIR,
            cls.VIDEO_POOL_DIR, cls.WAV_INPUT_DIR, cls.WAV_OUTPUT_DIR, cls.TS_OUTPUT_DIR,
            cls.PROCESSED_DIR, cls.LLM_DIR, cls.PLM_DIR, cls.PUBLIC_COMMENTS_DIR,
            cls.GENERATED_DIR, cls.TRAIN_DIR, cls.EVAL_DIR, cls.TEST_DIR,
            cls.PSL_LEARNT_WEIGHT_DIR, cls.PSL_TEMP_LEARN_DIR, cls.PSL_TEMP_INFER_DIR, 
            cls.PSL_OUTPUT_DIR
        ]:
            path.mkdir(parents=True, exist_ok=True) 