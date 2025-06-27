import sys
import time
import os
import json
import random
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as prfs
from collections import defaultdict
import operator

from pslpython.model import Model
from pslpython.partition import Partition
from pslpython.predicate import Predicate
from pslpython.rule import Rule

# Import configuration
from config import Settings, Paths

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

ADDITIONAL_PSL_OPTIONS = {
    'log4j.threshold': 'INFO'
}

ADDITIONAL_CLI_OPTIONS = [
    # '--postgres'
]

def main(args):
    seed_everything(args.seed)
   
    # Use configuration instead of global variables
    model_name = Settings.PSL_TRAIN_MODEL_NAME
    train_dir = Paths.PSL_GENERATED_TRAIN_DATA
    eval_dir = Paths.PSL_PROCESSED_TEST_DATA
    this_dir = Paths.PSL_TRAINING_DIR
    
    weight_directory = Paths.PSL_LEARNT_WEIGHT_DIR
    weight_file_loc = Paths.PSL_INIT_WEIGHT_FILE
    
    with open(weight_file_loc) as f:
        weight_file = json.load(f)
    
    model = Model(model_name)

    # Add Predicates
    add_predicates(model)

    # Add Rules
    add_rules(model, weight_file)
    
    # Model training to get the learnt weights
    learn(model, Paths.PSL_TEMP_LEARN_DIR, train_dir, eval_dir)

    # Write down the learnt weights
    write_weights(model, weight_directory)
    
    return True

def write_weights(model, weight_directory):
    weight_path = os.path.join(weight_directory, 'learnt_weights.txt')
    os.makedirs(weight_directory, exist_ok = True)
    with open(weight_path,'w') as f:
        for rule in model.get_rules():
            print('   ' + str(rule))
            f.write('   ' + str(rule) + '\n')

# Declare predicates for the model
def add_predicates(model):

    predicate = Predicate('LongUtterRatio', closed = True, size = 2)
    model.add_predicate(predicate)
    
    predicate = Predicate('HighCount', closed = True, size = 2)
    model.add_predicate(predicate)
    
    predicate = Predicate('LowCount', closed = True, size = 2)
    model.add_predicate(predicate)
    
    predicate = Predicate('Spoken', closed = True, size = 3)
    model.add_predicate(predicate)
    
    predicate = Predicate('CommentPhrase', closed = True, size = 2)
    model.add_predicate(predicate)
    
    predicate = Predicate('HearingPhrase', closed = True, size = 2)
    model.add_predicate(predicate)
    
    predicate = Predicate('NamePhrase', closed = True, size = 2)
    model.add_predicate(predicate)
    
    predicate = Predicate('Section', closed = False, size = 3)
    model.add_predicate(predicate)
    
    predicate = Predicate('SectionGPT', closed = True, size = 3)
    model.add_predicate(predicate)
    
    predicate = Predicate('Precedes', closed = True, size = 3)
    model.add_predicate(predicate)
    
    predicate = Predicate('First', closed = True, size = 2)
    model.add_predicate(predicate)   

    predicate = Predicate('SpeakerType', closed = False, size = 3)
    model.add_predicate(predicate)

    predicate = Predicate('CommentType', closed = False, size = 3)
    model.add_predicate(predicate)

    predicate = Predicate('CommentTypeLLM', closed = True, size = 3)
    model.add_predicate(predicate)


# Add rules and corresponding weights
def add_rules(model, weight_file):
    
  ######################  
        
    rules_list = [
        # speaker rules
        'LongUtterRatio(M, S) -> SpeakerType(M, S, \"public\") ^2', 
        'HighCount(M, S) -> SpeakerType(M, S, \"other\") ^2', 
        'LowCount(M, S) -> SpeakerType(M, S, \"public\") ^2', 
        # linguistic
        'CommentPhrase(M, U) -> Section(M, U, \"PC\") ^2', 
        'HearingPhrase(M, U) -> Section(M, U, \"PH\") ^2', 
        
        'NamePhrase(M, U) & Spoken(M,U,S) -> CommentType(M, U, \"PC\") ^2',
        'NamePhrase(M, U) & Spoken(M,U,S) -> CommentType(M, U, \"PH\") ^2',
        # AI
        'SectionGPT(M,U,\"PH\") & CommentTypeLLM(M,U,\"PC\") -> CommentType(M, U, \"PH\") ^2',
        'SectionGPT(M,U,\"PC\") & CommentTypeLLM(M,U,\"PH\") -> CommentType(M, U, \"PC\") ^2',
        
        'SectionGPT(M,U,\"Other\") -> Section(M, U, \"Other\") ^2',
        'SectionGPT(M,U,\"PC\") -> Section(M, U, \"PC\") ^2',
        'SectionGPT(M,U,\"PH\") -> Section(M, U, \"PH\") ^2'   ,

        'CommentTypeLLM(M,U,\"PH\")  -> CommentType(M, U, \"PH\") ^2',
        'CommentTypeLLM(M,U,\"PC\")  -> CommentType(M, U, \"PC\") ^2',
        'CommentTypeLLM(M,U,\"Other\") -> CommentType(M, U, \"Other\") ^2',
        # structural
        'First(M,U) -> Section(M, U, \"Other\") ^2',

        'Section(M, Uone, \"PC\") & Precedes(M,Uone,Utwo) -> Section(M, Utwo, \"PC\") ^2',
        'Section(M, Uone, \"PC\") & Precedes(M,Uone,Utwo) -> Section(M, Utwo, \"PH\") ^2',
        'Section(M, Uone, \"PC\") & Precedes(M,Uone,Utwo) -> Section(M, Utwo, \"Other\") ^2',

        'Section(M, Uone, \"PH\") & Precedes(M,Uone,Utwo) -> Section(M, Utwo, \"PC\") ^2',
        'Section(M, Uone, \"PH\") & Precedes(M,Uone,Utwo) -> Section(M, Utwo, \"PH\") ^2',
        'Section(M, Uone, \"PH\") & Precedes(M,Uone,Utwo) -> Section(M, Utwo, \"Other\") ^2',

        'Section(M, Uone, \"Other\") & Precedes(M,Uone,Utwo) -> Section(M, Utwo, \"PC\") ^2',
        'Section(M, Uone, \"Other\") & Precedes(M,Uone,Utwo) -> Section(M, Utwo, \"PH\") ^2',
        'Section(M, Uone, \"Other\") & Precedes(M,Uone,Utwo) -> Section(M, Utwo, \"Other\") ^2',

        'Section(M,Uone,\"PC\")& Section(M,Utwo,\"PC\")&CommentType(M, Uone, \"Other\") & Precedes(M,Uone,Utwo) -> CommentType(M, Utwo, \"PC\") ^2',
        'Section(M,Uone,\"PH\")& Section(M,Utwo,\"PH\")&CommentType(M, Uone, \"Other\") & Precedes(M,Uone,Utwo) -> CommentType(M, Utwo, \"PH\") ^2',

        'Section(M, U, \"PC\") & Spoken(M,U,S)  & SpeakerType(M, S, \"public\") -> CommentType(M, U, \"PC\") ^2',
        'Section(M, U, \"PC\") & Spoken(M,U,S)  & SpeakerType(M, S, \"other\") -> CommentType(M, U, \"Other\") ^2',
        'Section(M, U, \"PH\") & Spoken(M,U,S)  & SpeakerType(M, S, \"public\") -> CommentType(M, U, \"PH\") ^2',
        'Section(M, U, \"PH\") & Spoken(M,U,S)  & SpeakerType(M, S, \"other\") -> CommentType(M, U, \"Other\") ^2',
        # constraints
        '!SpeakerType(M, S, \"public\") ^2',
    ]
    # constraints
    model.add_rule(Rule('SpeakerType(M, S,+d) = 1 .'))
    model.add_rule(Rule('CommentType(M, U,+d) = 1 .'))
    model.add_rule(Rule('Section(M, U,+d) = 1 .'))
    
    for rule in rules_list:
        model.add_rule(Rule(str(weight_file[rule]) + ": " + rule))

# Load data from files
def add_data(model, train_type, train_dir, eval_dir):
    for predicate in model.get_predicates().values():
        predicate.clear_data()
        
    DATA_DIR = train_dir if train_type == "train" else eval_dir
    
    path = os.path.join(DATA_DIR, 'spoken.txt')
    model.get_predicate('Spoken').add_data_file(Partition.OBSERVATIONS, path)

    path = os.path.join(DATA_DIR, 'speaker_count_high.txt')
    model.get_predicate('HighCount').add_data_file(Partition.OBSERVATIONS, path)

    path = os.path.join(DATA_DIR, 'speaker_count_low.txt')
    model.get_predicate('LowCount').add_data_file(Partition.OBSERVATIONS, path)
    
    path = os.path.join(DATA_DIR, 'comment_phrase.txt')
    model.get_predicate('CommentPhrase').add_data_file(Partition.OBSERVATIONS, path)
    
    path = os.path.join(DATA_DIR, 'hearing_phrase.txt')
    model.get_predicate('HearingPhrase').add_data_file(Partition.OBSERVATIONS, path)
    
    path = os.path.join(DATA_DIR, 'name_phrase.txt')
    model.get_predicate('NamePhrase').add_data_file(Partition.OBSERVATIONS, path)
    
    path = os.path.join(DATA_DIR, 'speaker_long.txt')
    model.get_predicate('LongUtterRatio').add_data_file(Partition.OBSERVATIONS, path)

    path = os.path.join(DATA_DIR, 'precedes.txt')
    model.get_predicate('Precedes').add_data_file(Partition.OBSERVATIONS, path)
    
    path = os.path.join(DATA_DIR, 'first.txt')
    model.get_predicate('First').add_data_file(Partition.OBSERVATIONS, path)

    path = os.path.join(DATA_DIR, 'sectiontype_obs.txt')
    model.get_predicate('SectionGPT').add_data_file(Partition.OBSERVATIONS, path)

    path = os.path.join(DATA_DIR, 'commenttype_target.txt')
    model.get_predicate('CommentType').add_data_file(Partition.TARGETS, path)

    path = os.path.join(DATA_DIR, 'commenttype_truth.txt')
    model.get_predicate('CommentType').add_data_file(Partition.TRUTH, path)
    
    path = os.path.join(DATA_DIR, 'commenttype_llm.txt')
    model.get_predicate('CommentTypeLLM').add_data_file(Partition.OBSERVATIONS, path)
    
    path = os.path.join(DATA_DIR, 'speaker_type_target.txt')
    model.get_predicate('SpeakerType').add_data_file(Partition.TARGETS, path)

    path = os.path.join(DATA_DIR, 'speaker_type_truth.txt')
    model.get_predicate('SpeakerType').add_data_file(Partition.TRUTH, path)

    path = os.path.join(DATA_DIR, 'sectiontype_target.txt')
    model.get_predicate('Section').add_data_file(Partition.TARGETS, path)

    path = os.path.join(DATA_DIR, 'sectiontype_truth.txt')
    model.get_predicate('Section').add_data_file(Partition.TRUTH, path)
    
def learn(model, temp_dir, train_dir, eval_dir):
    add_data(model, 'train', train_dir, eval_dir)
    model.learn(temp_dir = temp_dir,additional_cli_options = ADDITIONAL_CLI_OPTIONS, psl_config = ADDITIONAL_PSL_OPTIONS)
