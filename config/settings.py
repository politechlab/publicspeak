class Settings:
    # model settings
    MODEL_NAME = "roberta-large"
    DEVICE = "cuda"
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    EPOCHS = 7
    SEED = 42
    
    # video processing settings
    VIDEO_MODE = "full"
    
    # whisperX settings
    WHISPER_MODEL_NAME = "large-v3"
    WHISPER_DEVICE = "cuda:1,2,3,4,5,6,7"
    WHISPER_LANGUAGE_CODE = "en"
    WHISPER_BATCH_SIZE = 36
    WHISPER_COMPUTE_TYPE = "float16"
    
    # trigger settings
    LONG_TEXT_THRESHOLD = 50
    RATIO_COUNT = 0.5
    CUT_OFF_THRESHOLD = 50
    GPT_VERSION = "gpt-4"
    
    # data processing settings
    CITY = "AA"
    
    # signal words settings
    HEARING_SIGNALS = ["open up the public hearing", "public hearing"]
    COMMENT_SIGNALS = ["public comment"]
    NEXT_SIGNALS = ["next speaker"]
    NAME_SIGNALS = ["my name is"]
    
    # comment type mapping
    COMMENT_TYPE_MAPPING = ["Other", "PC", "PH"]
    
    # file naming settings
    PLM_BATCH_SIZE = 8
    PLM_PRED_FILE = f"{CITY}_pred_LOO_roberta.json"
    TRAIN_FILE = f"{CITY}_train.json"
    VAL_FILE = f"{CITY}_val.json"
    TEST_FILE = f"{CITY}_test.json" 