# publicspeak


# data -
- full-comment-data-with-topics.csv is a file for generating the topic distribution figures
- 
- 

# pipeline 
 - transcribe_and_LLM contains the code to transcribe the mp4 files and generate LLM indicators
 - PLM the code here generates PLM predictions which are used by the PSL model
 - generate_processed_data contains a script for transforming all of the data into the format that PSL can use 

# model 
- PSL code
    - training - trains a PSL model and uses the model to make inferences 
    - inference - generates the results discussed in the paper  

# experiments 
- the code to reproduce results in paper

# evaluation and analysis - 
- a notebook for generating topic assignments and delete the key - Eva
- a folder with prompts
- a script for reading from results and generating metrics 
