# PublicSpeak: 
**PUBLICSPEAK: Hearing the Public with a Probabilistic Framework**

---
Local governments around the world are making consequential decisions on behalf of their constituents, and these constituents are responding with requests, advice, and assessments of their officials at public meetings. So many small meetings cannot be covered by traditional newsrooms at scale. We propose PUBLICSPEAK, a probabilistic framework which can utilize meeting structure, domain knowledge, and linguistic information to discover public remarks in local government meetings. We then use our approach to inspect the issues raised by constituents in 7 cities across the United States. We evaluate our approach on a novel dataset of local government meetings and find that PUBLICSPEAK improves over state-of-the-art by 10% on average, and by up to 40%. 

## Installation

**Stable Release:** `pip install publicspeak`<br>

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
