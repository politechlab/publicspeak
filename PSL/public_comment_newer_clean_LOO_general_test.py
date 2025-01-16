import sys
import time
import os
import pickle
import json
import random
import numpy as np
from pslpython.model import Model
from pslpython.partition import Partition
from pslpython.predicate import Predicate
from pslpython.rule import Rule

#MODEL_NAME = 'simple-pc-ll-aa'
#FOLD = '4'
#THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
#TRAIN_DIR = os.path.join(THIS_DIR, '..', 'data', 'public_comments_section_version', 'train', FOLD)
#EVAL_DIR = os.path.join(THIS_DIR, '..', 'data', 'public_comments_section_version', 'eval', FOLD)
#DATA_DIR = os.path.join(THIS_DIR, '..', 'data', 'public_comments', '0', 'eval', 'cv', '0')
#print(THIS_DIR)

#weight_grid = [{'speaker_prior':'1','gpt_prior':1},{'speaker_prior':'5','gpt_prior':5},{'speaker_prior':'10','gpt_prior':10},{'speaker_prior':'20','gpt_prior':20},{'speaker_prior':'25','gpt_prior':25},{'speaker_prior':'30','gpt_prior':50},{'speaker_prior':'15','gpt_prior':100}]
#weight_grid = [{'speaker_prior':'15','gpt_prior':100,'first_prior':1,'same_prior':10,'speaker_prior2':'25'},{'speaker_prior':'15','gpt_prior':100,'first_prior':5,'same_prior':10,'speaker_prior2':'1'},{'speaker_prior':'15','gpt_prior':100,'first_prior':1,'same_prior':'10','speaker_prior2':5},{'speaker_prior':'15','gpt_prior':100,'first_prior':1,'same_prior':10,'speaker_prior2':10}]

with open("hyperparameter/weight_grid_clean.p" ,'rb') as f:
        weight_grid = pickle.load(f)
        
# weight_grid = [{'speaker_prior': 0,
#   'gpt_prior': 0,
#   'first_prior': 0,
#   'section_prior': 0,
#   'alt_prior': 0,
#   'alt_prior2': 0,
#   'other_prior': 0,
#   'low_count': 0,
#   'llm_prior': 0,
#   'llm_prior2': 0,
#   'name_prior': 0,
#   'word_prior': 0,
#   'same_prior': 0,
#   'hearing_prior': 0}]
def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    # torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed_value)
    #     torch.cuda.manual_seed_all(seed_value)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = True

seed_everything(42)

ADDITIONAL_PSL_OPTIONS = {
    'log4j.threshold': 'INFO'
}

ADDITIONAL_CLI_OPTIONS = [
    # '--postgres'
]

def main(fold, directory, grid_value, city, name):
   
    global FOLD
    FOLD = fold 
       
    global THIS_DIR
    global TRAIN_DIR
    global EVAL_DIR
    global DATA_DIR
    global MODEL_NAME
    MODEL_NAME = "base_clean_{}_{}_{}_{}".format(city, name, fold, grid_value)
    model = Model(MODEL_NAME)

    THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    TRAIN_DIR = os.path.join(THIS_DIR, '..', 'data', directory, 'train', FOLD)
    EVAL_DIR = os.path.join(THIS_DIR, '..', 'data', directory, 'test', FOLD)
    

    model = Model(MODEL_NAME)

    # Add Predicates
    add_predicates(model)

    # Add Rules
    add_rules(model,grid_value)
    
    # Train
    print("here")
    learn(model)
    print("here1")
    with open('learned_rules/write_rules_clean_{}_test.txt'.format(MODEL_NAME),'a') as f:
        for rule in model.get_rules():
            print('   ' + str(rule))
            f.write('   ' + str(rule)+'\n')
    # Inference
    results = infer(model)

    write_results(results, model,grid_value,MODEL_NAME[:-2],city,name)
    
    return True

def write_results(results, model,grid_value,model_name,city,name):
    out_dir = os.path.join(THIS_DIR, 'inferred-predicates_LOO_{}_{}_{}_test'.format(name,model_name.split(city)[0][:-1],city))
    os.makedirs(out_dir, exist_ok = True)
    print(out_dir)
    for predicate in model.get_predicates().values():
        if (predicate.closed()):
            continue

        out_path = os.path.join(out_dir, "%s-%s-%s-%s.txt" % (predicate.name(),city,FOLD,str(grid_value)))
        results[predicate].to_csv(out_path, sep = "\t", header = False, index = False)

        
def add_predicates(model):

    #predicate = Predicate('SectionType', closed = False, size = 2)
    #model.add_predicate(predicate)
    
    predicate = Predicate('LongUtterRatio', closed = True, size = 2)
    model.add_predicate(predicate)
    
    predicate = Predicate('IsLongUtter', closed = True, size = 2)
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
    
    predicate = Predicate('NextPhrase', closed = True, size = 2)
    model.add_predicate(predicate)
    
    
    predicate = Predicate('Section', closed = False, size = 3)
    model.add_predicate(predicate)
    
    predicate = Predicate('SectionGPT', closed = True, size = 3)
    model.add_predicate(predicate)
    
    predicate = Predicate('Precedes', closed = True, size = 3)
    model.add_predicate(predicate)
    
    predicate = Predicate('First', closed = True, size = 2)
    model.add_predicate(predicate)
    
    
    predicate = Predicate('Contain', closed = True, size = 3)
    model.add_predicate(predicate)

    predicate = Predicate('SpeakerType', closed = False, size = 3)
    model.add_predicate(predicate)

    predicate = Predicate('CommentType', closed = False, size = 3)
    model.add_predicate(predicate)

    predicate = Predicate('CommentTypeLLM', closed = True, size = 3)
    model.add_predicate(predicate)

    
#     predicate = Predicate('CommentKeywordsRatio', closed = True, size = 1)
#     model.add_predicate(predicate)
    
#     predicate = Predicate('HearingKeywordsRatio', closed = True, size = 1)
#     model.add_predicate(predicate)

def add_rules(model,grid_value):
    
  ######################  
#speaker rules
    model.add_rule(Rule('10: LongUtterRatio(M, S) -> SpeakerType(M, S, \"public\") ^2'))

    model.add_rule(Rule('10: HighCount(M, S) -> SpeakerType(M, S, \"other\") ^2'))
    model.add_rule(Rule('{}: LowCount(M, S) -> SpeakerType(M, S, \"public\") ^2'.format(weight_grid[grid_value]['low_count'])))

##########################
#linguistic

    model.add_rule(Rule('{}: CommentPhrase(M, U) -> Section(M, U, \"PC\") ^2'.format(weight_grid[grid_value]["word_prior"])))
    model.add_rule(Rule('{}: HearingPhrase(M, U) -> Section(M, U, \"PH\") ^2'.format(weight_grid[grid_value]["hearing_prior"])))

    model.add_rule(Rule('{}: NamePhrase(M, U) & Spoken(M,U,S) -> CommentType(M, U, \"PC\") ^2'.format(weight_grid[grid_value]["name_prior"])))
    model.add_rule(Rule('{}: NamePhrase(M, U) & Spoken(M,U,S) -> CommentType(M, U, \"PH\") ^2'.format(weight_grid[grid_value]["name_prior"])))
    #model.add_rule(Rule('{}: CommentTypeLLM(M,U,\"PC\") & Section(M, U, \"PH\")-> CommentType(M, U, \"PH\") ^2'.format(weight_grid[grid_value]["llm_prior2"])))
    
    #model.add_rule(Rule('{}: CommentTypeLLM(M,U,\"PH\") & Section(M, U, \"PC\")-> CommentType(M, U, \"PC\") ^2'.format(weight_grid[grid_value]["llm_prior2"])))
    
    #model.add_rule(Rule('{}: CommentTypeLLM(M,U,\"PC\") & Section(M, U, \"Other\")-> CommentType(M, U, \"Other\") ^2'.format(weight_grid[grid_value]["llm_prior2"])))
    
    #model.add_rule(Rule('{}: CommentTypeLLM(M,U,\"PH\") & Section(M, U, \"Other\")-> CommentType(M, U, \"Other\") ^2'.format(weight_grid[grid_value]["llm_prior2"])))
    
  ###################
#AI
    # model.add_rule(Rule('{}: SectionGPT(M,U,\"PH\") & CommentTypeLLM(M,U,\"PC\") -> CommentType(M, U, \"PH\")  ^2'.format(weight_grid[grid_value]["alt_prior"])))
    model.add_rule(Rule('{}: SectionGPT(M,U,\"PH\") & CommentTypeLLM(M,U,\"PC\") -> CommentType(M, U, \"PH\")  ^2'.format(weight_grid[grid_value]["alt_prior"])))
    
    # This for RCH!  
    model.add_rule(Rule('50: SectionGPT(M,U,\"PC\") & CommentTypeLLM(M,U,\"PH\") -> CommentType(M, U, \"PC\")  ^2'))
    #model.add_rule(Rule('{}: SectionGPT(M,U,\"PC\") & CommentTypeLLM(M,U,\"PH\") -> CommentType(M, U, \"PC\")  ^2'.format(weight_grid[grid_value]["alt_prior"])))

    model.add_rule(Rule('{}: SectionGPT(M,U,\"Other\") -> Section(M, U, \"Other\") ^2'.format(weight_grid[grid_value]["gpt_prior"])))
    model.add_rule(Rule('{}: SectionGPT(M,U,\"PC\") -> Section(M, U, \"PC\") ^2'.format(weight_grid[grid_value]["gpt_prior"])))
    model.add_rule(Rule('{}: SectionGPT(M,U,\"PH\") -> Section(M, U, \"PH\") ^2'.format(weight_grid[grid_value]["gpt_prior"])))
    # model.add_rule(Rule('{}: SectionGPT(M,U,\"PH\") -> Section(M, U, \"PH\") ^2'.format(weight_grid[grid_value]["alt_prior"])))
    
    
    model.add_rule(Rule('{}: CommentTypeLLM(M,U,\"PH\")  -> CommentType(M, U, \"PH\") ^2'.format(weight_grid[grid_value]["llm_prior"])))
    model.add_rule(Rule('{}: CommentTypeLLM(M,U,\"PC\")  -> CommentType(M, U, \"PC\") ^2'.format(weight_grid[grid_value]["llm_prior"])))
    
    model.add_rule(Rule('{}: CommentTypeLLM(M,U,\"Other\") -> CommentType(M, U, \"Other\") ^2'.format(weight_grid[grid_value]["llm_prior"])))
           
        
#######################################

#structural
    model.add_rule(Rule('{}: First(M,U) -> Section(M, U, \"Other\") ^2'.format(weight_grid[grid_value]["first_prior"])))
    
    model.add_rule(Rule('{}: Section(M, Uone, \"PC\") & Precedes(M,Uone,Utwo) -> Section(M, Utwo, \"PC\") ^2'.format(weight_grid[grid_value]["same_prior"])))
    model.add_rule(Rule('10: Section(M, Uone, \"PC\") & Precedes(M,Uone,Utwo) -> Section(M, Utwo, \"PH\") ^2'))
    model.add_rule(Rule('10: Section(M, Uone, \"PC\") & Precedes(M,Uone,Utwo) -> Section(M, Utwo, \"Other\") ^2'))



    model.add_rule(Rule('10: Section(M, Uone, \"PH\") & Precedes(M,Uone,Utwo) -> Section(M, Utwo, \"PC\") ^2'))
    model.add_rule(Rule('{}: Section(M, Uone, \"PH\") & Precedes(M,Uone,Utwo) -> Section(M, Utwo, \"PH\") ^2'.format(weight_grid[grid_value]["same_prior"])))
    model.add_rule(Rule('10: Section(M, Uone, \"PH\") & Precedes(M,Uone,Utwo) -> Section(M, Utwo, \"Other\") ^2'))


    model.add_rule(Rule('10: Section(M, Uone, \"Other\") & Precedes(M,Uone,Utwo) -> Section(M, Utwo, \"PC\") ^2'))
    model.add_rule(Rule('10: Section(M, Uone, \"Other\") & Precedes(M,Uone,Utwo) -> Section(M, Utwo, \"PH\") ^2'))
    model.add_rule(Rule('{}: Section(M, Uone, \"Other\") & Precedes(M,Uone,Utwo) -> Section(M, Utwo, \"Other\") ^2'.format(weight_grid[grid_value]["same_prior"])))
    
    model.add_rule(Rule('{}: Section(M,Uone,\"PC\")& Section(M,Utwo,\"PC\")&CommentType(M, Uone, \"Other\") & Precedes(M,Uone,Utwo) -> CommentType(M, Utwo, \"PC\") ^2'.format(weight_grid[grid_value]["alt_prior2"])))
    
    model.add_rule(Rule('{}: Section(M,Uone,\"PH\")& Section(M,Utwo,\"PH\")&CommentType(M, Uone, \"Other\") & Precedes(M,Uone,Utwo) -> CommentType(M, Utwo, \"PH\") ^2'.format(weight_grid[grid_value]["alt_prior2"])))
    
#     model.add_rule(Rule('0: Section(M,Uone,\"PH\")-> !CommentType(M, Uone, \"PC\") ^2'))
    
#     model.add_rule(Rule('0: Section(M,Uone,\"PC\")-> !CommentType(M, Uone, \"PH\") ^2'))

     ###################
    model.add_rule(Rule('{}: Section(M, U, \"PC\") & Spoken(M,U,S)  & SpeakerType(M, S, \"public\") -> CommentType(M, U, \"PC\") ^2'.format(weight_grid[grid_value]['section_prior'])))

    model.add_rule(Rule('{}: Section(M, U, \"PC\") & Spoken(M,U,S)  & SpeakerType(M, S, \"other\") -> CommentType(M, U, \"Other\") ^2'.format(weight_grid[grid_value]['section_prior'])))
    model.add_rule(Rule('{}: Section(M, U, \"PH\") & Spoken(M,U,S)  & SpeakerType(M, S, \"public\") -> CommentType(M, U, \"PH\") ^2'.format(weight_grid[grid_value]['section_prior'])))

    model.add_rule(Rule('{}: Section(M, U, \"PH\") & Spoken(M,U,S)  & SpeakerType(M, S, \"other\") -> CommentType(M, U, \"Other\") ^2'.format(weight_grid[grid_value]['section_prior'])))
    
    '''
    manager role
    '''
      
    model.add_rule(Rule('SpeakerType(M, S,+d) = 1 .'))
    model.add_rule(Rule('CommentType(M, U,+d) = 1 .'))
    model.add_rule(Rule('Section(M, U,+d) = 1 .'))

    model.add_rule(Rule('{}: !SpeakerType(M, S, \"public\") ^2'.format(weight_grid[grid_value]['speaker_prior'])))

def add_data(model, train_type):
    for predicate in model.get_predicates().values():
        predicate.clear_data()
        
    DATA_DIR = TRAIN_DIR if train_type == "train" else EVAL_DIR
    
    
    path = os.path.join(DATA_DIR, 'spoken.txt')
    model.get_predicate('Spoken').add_data_file(Partition.OBSERVATIONS, path)

    path = os.path.join(DATA_DIR, 'speaker_count_high.txt')
    model.get_predicate('HighCount').add_data_file(Partition.OBSERVATIONS, path)

    path = os.path.join(DATA_DIR, 'speaker_count_low.txt')
    model.get_predicate('LowCount').add_data_file(Partition.OBSERVATIONS, path)

    path = os.path.join(DATA_DIR, 'contains.txt')
    model.get_predicate('Contain').add_data_file(Partition.OBSERVATIONS, path)
    
    path = os.path.join(DATA_DIR, 'comment_phrase.txt')
    model.get_predicate('CommentPhrase').add_data_file(Partition.OBSERVATIONS, path)
    
    path = os.path.join(DATA_DIR, 'hearing_phrase.txt')
    model.get_predicate('HearingPhrase').add_data_file(Partition.OBSERVATIONS, path)
    
    path = os.path.join(DATA_DIR, 'next_phrase.txt')
    model.get_predicate('NextPhrase').add_data_file(Partition.OBSERVATIONS, path)
    
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

    path = os.path.join(DATA_DIR, 'longUtter.txt')
    model.get_predicate('IsLongUtter').add_data_file(Partition.OBSERVATIONS, path)

    path = os.path.join(DATA_DIR, 'commenttype_target.txt')
    model.get_predicate('CommentType').add_data_file(Partition.TARGETS, path)

    path = os.path.join(DATA_DIR, 'commenttype_truth.txt')
    model.get_predicate('CommentType').add_data_file(Partition.TRUTH, path)
    
    path = os.path.join(DATA_DIR, 'commenttype_llm.txt')
    model.get_predicate('CommentTypeLLM').add_data_file(Partition.OBSERVATIONS, path)

#     path = os.path.join(DATA_DIR, 'speaker_target_sabina.txt')
#     model.get_predicate('SpeakerType').add_data_file(Partition.TARGETS, path)

#     path = os.path.join(DATA_DIR, 'speaker_truth_sabina.txt')
#     model.get_predicate('SpeakerType').add_data_file(Partition.TRUTH, path)
    
    path = os.path.join(DATA_DIR, 'speaker_type_target.txt')
    model.get_predicate('SpeakerType').add_data_file(Partition.TARGETS, path)

    path = os.path.join(DATA_DIR, 'speaker_type_truth.txt')
    model.get_predicate('SpeakerType').add_data_file(Partition.TRUTH, path)

    path = os.path.join(DATA_DIR, 'sectiontype_target.txt')
    model.get_predicate('Section').add_data_file(Partition.TARGETS, path)

    path = os.path.join(DATA_DIR, 'sectiontype_truth.txt')
    model.get_predicate('Section').add_data_file(Partition.TRUTH, path)

def learn(model):
    add_data(model,'train')
    model.learn(temp_dir = "temp0",additional_cli_options = ADDITIONAL_CLI_OPTIONS, psl_config = ADDITIONAL_PSL_OPTIONS)    

def infer(model):
    add_data(model,'test')
    return model.infer(temp_dir = "temp0",additional_cli_options = ADDITIONAL_CLI_OPTIONS, psl_config = ADDITIONAL_PSL_OPTIONS)


if (__name__ == '__main__'):

    start = int(sys.argv[1])
    end = int(sys.argv[2])

    # ds = ["public_comments_AA_LOO", "public_comments_RO_LOO", "public_comments_JS_LOO", 
    #      "public_comments_LS_LOO", "public_comments_AP_LOO", "public_comments_IN_LOO", "public_comments_PE_LOO", 
    #      "public_comments_SEA_LOO", "public_comments_OAK_LOO", ]
    # cs = ["AA", "RO", "JS", 
    #      "LS", "AP", "IN", "PE", 
    #      "SEA", "OAK", ]
    # param = [0, 0, 0, 
    #         0, 0, 0, 0, 
    #         0, 0]

    # ds = ["public_comments_AA_LOO_general_prompt", "public_comments_RO_LOO_general_prompt", "public_comments_JS_LOO_general_prompt", 
    #      "public_comments_LS_LOO_general_prompt", 
    #      "public_comments_SEA_LOO_general_prompt", "public_comments_OAK_LOO_general_prompt", "public_comments_RCH_LOO_general_prompt"]
    # cs = ["AA", "RO", "JS", 
    #      "LS", 
    #      "SEA", "OAK", "RCH", ]
    ds = ["public_comments_LS_LOO_general_prompt"]
    cs = ["LS"]
    param = [0, 0, 0, 0, 0, 0, 0]
    for i in range(len(ds)):
        print(param[i], param[i] + 1)
        for j in range(param[i], param[i] + 1):
            for fold in ['0']:
                while(True):
                    try:
                        name = 'general'
                        main(fold,ds[i],j,cs[i],name)
                        break
                    except Exception as e:
                        print(e)
                        break
                        continue
