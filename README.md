# publicspeak


# data - PUTTING THE SMALL DATA IN IS HIGH PRIORITY 
- link to much data - publish in zenodo, dataverse
- small data for experiments

# pipeline - this is second priority 
 - video to audio
 - transcript
 - clean transcript
 - feed the transcript to PLM
 - feed the transcript to LLM
 - give transcript, PLM output, LLM output to PSL 

# model - HIGH PRIORITY
- PSL code
    - training - takes data from the pipeline
    - inference - start with inference - 
       - one for the paper experiments, use the best weights learned
       - with default weights
       - let's keep track of the size of the models, if weights are too large we can put in zenodo
       - write results to results folder
         


# experiments - HIGH PRIORITY FOR THE PSL RESULTS

- the code to reproduce results in paper

# evaluation and analysis - 
- Sabina add notebook for plotting
- a notebook for generating topic assignments and delete the key
- a folder with prompts
- a script for reading from results and generating metrics 
