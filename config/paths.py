from pathlib import Path

class Paths:
    # base path
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    
    # raw data path
    RAW_DIR = DATA_DIR / "raw"
    AUDIO_DIR = RAW_DIR / "audio"
    VIDEO_DIR = RAW_DIR / "video"
    TRANSCRIPTS_DIR = RAW_DIR / "transcripts"
    
    # video and audio processing paths
    VIDEO_POOL_DIR = DATA_DIR / "video_pool"
    WAV_INPUT_DIR = AUDIO_DIR / "input"
    WAV_OUTPUT_DIR = AUDIO_DIR / "output"
    TS_OUTPUT_DIR = TRANSCRIPTS_DIR / "new_ts_output"
    
    # processed data path
    PROCESSED_DIR = DATA_DIR / "processed"
    LLM_DIR = PROCESSED_DIR / "LLM_indicators"
    PLM_DIR = PROCESSED_DIR / "PLM_indicators"
    PUBLIC_COMMENTS_DIR = PROCESSED_DIR / "public_comments"
    
    # generated data path
    GENERATED_DIR = DATA_DIR / "generated"
    TRAIN_DIR = GENERATED_DIR / "train"
    EVAL_DIR = GENERATED_DIR / "eval"
    TEST_DIR = GENERATED_DIR / "test"
    
    @classmethod
    def create_dirs(cls):
        """create all necessary directories"""
        for path in [
            cls.RAW_DIR, cls.AUDIO_DIR, cls.VIDEO_DIR, cls.TRANSCRIPTS_DIR,
            cls.VIDEO_POOL_DIR, cls.WAV_INPUT_DIR, cls.WAV_OUTPUT_DIR, cls.TS_OUTPUT_DIR,
            cls.PROCESSED_DIR, cls.LLM_DIR, cls.PLM_DIR, cls.PUBLIC_COMMENTS_DIR,
            cls.GENERATED_DIR, cls.TRAIN_DIR, cls.EVAL_DIR, cls.TEST_DIR
        ]:
            path.mkdir(parents=True, exist_ok=True) 