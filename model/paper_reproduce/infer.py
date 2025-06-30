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

def main(args, city):
    seed_everything(args.seed)
   
    # Use configuration instead of global variables
    model_name = Settings.PSL_TEST_MODEL_NAME
    eval_dir = Paths.PSL_PROCESSED_TEST_DATA
    this_dir = Paths.PSL_PAPER_REPRODUCE_DIR
    
    weight_file_loc = Paths.PSL_WEIGHT_FILE
    
    with open(weight_file_loc) as f:
        weight_file = json.load(f)
    
    model = Model(model_name)
    
    # Add Predicates
    add_predicates(model)
    
    # Add Rules
    add_rules(model, weight_file, city)

    # Model infers to get results
    results = infer(model, Paths.PSL_PAPER_REPRODUCE_OUTPUT, eval_dir, city)

    # Write down the results
    write_results(results, model, Paths.PSL_PAPER_REPRODUCE_OUTPUT, city)
    
    return True

# Write results to a folder
def write_results(results, model, output_directory, city):
    os.makedirs(output_directory, exist_ok = True)
    for predicate in model.get_predicates().values():
        if (predicate.closed()):
            continue

        out_path = os.path.join(output_directory, f"{city}_pred.txt")
        results[predicate].to_csv(out_path, sep = "\t", header = False, index = False)

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
def add_rules(model, weight_file, city):
    
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
        model.add_rule(Rule(weight_file[city][rule] + ": " + rule))

# Load data from files
def add_data(model, eval_dir):
    for predicate in model.get_predicates().values():
        predicate.clear_data()
        
    DATA_DIR = eval_dir
    
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
    
def infer(model, temp_dir, eval_dir, city):
    add_data(model, eval_dir / city)
    return model.infer(temp_dir = temp_dir, additional_cli_options = ADDITIONAL_CLI_OPTIONS, psl_config = ADDITIONAL_PSL_OPTIONS)

# Get predictions
def get_pred(city, output_directory):
    with open(f"{output_directory}/{city}_pred.txt") as f:        
        pred = f.readlines()
    return pred

def get_pred_list(city, output_directory):
    pred = get_pred(city, output_directory)

    pred_l = []
    
    to_return = {}
    for p_i in pred:
        x = p_i.strip().split("\t")
        meet_id = x[0]
        ut_id = x[1]
        ct = x[2]
        tv = x[3]
        
        key = '{}-{}'.format(meet_id,ut_id)
        
        if key not in to_return:
            to_return[key] = {'PH':0,'PC':0,'Other':0}
        to_return[key][ct]=float(tv)
    return {k:sorted(v.items(), key=operator.itemgetter(1),reverse=True)[0][0] for k,v in to_return.items()}     

# Get label files
def get_truth_modified(eval_dir, city):
    with open(eval_dir / f"{city}/commenttype_truth.txt") as f:         
        truth = f.readlines()
    return truth

def get_truth_dict_modified(eval_dir, city):
    truth = get_truth_modified(eval_dir, city)
    to_return = {}
    for line in truth:
        x = line.strip('\n').split('\t')
        meet_id = x[0]
        ut_id = x[1]
        ct = x[2]
        tv = x[3]
        key = '{}-{}'.format(meet_id,ut_id)
        
        if key not in to_return:
            to_return[key] = {'PH':0,'PC':0,'Other':0}
        to_return[key][ct]=float(tv)
    return {k:sorted(v.items(), key=operator.itemgetter(1),reverse=True)[0][0] for k,v in to_return.items()}

# Compute recall, precision and f1 for one target
def get_prfs(truth,preds,target):
    keys = sorted(truth.keys())
    true_vec = [int(truth[k]==target) for k in keys]
    pred_vec = [int(preds[k]==target) for k in keys]
    return true_vec,pred_vec

def all_(target, output_directory, eval_dir, city):
    all_true = []
    all_pred = []
    truth_dict = get_truth_dict_modified(eval_dir, city)
    preds = get_pred_list(city, output_directory)

    tv,pv = get_prfs(truth_dict, preds, target)
    all_true.extend(tv)
    all_pred.extend(pv)
    if len(set(all_true)) == 1 and len(set(all_pred)) == 1:
        return (None, None, None, None)
    return prfs(all_true, all_pred, average='binary')

def test(args, city):
    # Compute metric values for PC and PH
    print("here")
    eval_dir = Paths.PSL_PROCESSED_TEST_DATA
    output_directory = Paths.PSL_PAPER_REPRODUCE_OUTPUT
    #output_directory = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__))), output_directory)
    rs4 = all_('PC', output_directory, eval_dir, city)
    rs2 = all_('PH', output_directory, eval_dir, city)
    
    def a(s):
        if s is None:
            return "N/A"
        return "{:.3f}".format(round(s, 3))
    k = (rs4[2] + rs2[2]) / 2 if rs2[2] is not None else None
    out = f"Recall, Precision and F1 score of {city} Public Comments are: " + " & ".join([a(rs4[1]), a(rs4[0]), a(rs4[2])])
    print("=========================================================================")
    print(out)
    print("=========================================================================")
    return rs4, rs2
