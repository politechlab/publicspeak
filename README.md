# PublicSpeak: 
**PUBLICSPEAK: Hearing the Public with a Probabilistic Framework**

[PUBLICSPEAK: Hearing the Public with a Probabilistic Framework](https://www.researchgate.net/profile/Tianliang-Xu/publication/389691299_PUBLICSPEAK_Hearing_the_Public_with_a_Probabilistic_Framework/links/67cddaa132265243f5843511/PUBLICSPEAK-Hearing-the-Public-with-a-Probabilistic-Framework.pdf)

---
Local governments around the world are making consequential decisions on behalf of their constituents, and these constituents are responding with requests, advice, and assessments of their officials at public meetings. So many small meetings cannot be covered by traditional newsrooms at scale. We propose PUBLICSPEAK, a probabilistic framework which can utilize meeting structure, domain knowledge, and linguistic information to discover public remarks in local government meetings. We then use our approach to inspect the issues raised by constituents in 7 cities across the United States. We evaluate our approach on a novel dataset of local government meetings and find that PUBLICSPEAK improves over state-of-the-art by 10% on average, and by up to 40%. 

## Installation
- First, create a virtual environment
  
    `conda create --name publicspeak python=3.11 -y`
  
    `conda activate publicspeak`
- Install `just` software.

    `pip install rust-just`

- Run 
    `just install-pipeline-deps`
    to install all the dependencies needed to prepare for the data from the scratch.

- Run 
    `just setup-psl-env`
    to create virtual environment to run the experiments and training.

- Run
    `conda activate psl-global`
    to activiate the virtual environment psl-global.

### Installation Demo Video: click to view
[![PUBLICSPEAK Code Installation](https://img.youtube.com/vi/NdTGXwOtWtw/0.jpg)](https://www.youtube.com/watch?v=NdTGXwOtWtw)

## How to use

We provided several script to go through the project:
- **experiment.sh**: A script to replicate the results in our paper.

    Run 
    
    `bash experiment.sh`
    
- **training.sh**: A script to train your model.

    Change the parameters in `training.sh`
    
    Run 
    
    `bash training.sh`
    
---
We also provide scripts for preparing for the data.
- **transcribe_video.sh**: A script to transcribe a council meeting video.

    Change the parameters in `transcribe_video.sh`
    
    Run 
    
    `bash transcribe_video.sh`
    
- **prepare_publicspeak_data.sh**: A script to replicate the results in our paper.

    Change the parameters in `prepare_publicspeak_data.sh`
    
    Run 
    
    `bash prepare_publicspeak_data.sh`

## Pipeline 
 - transcribe_and_LLM contains the code to transcribe the mp4 files and generate LLM indicators
 - PLM the code here generates PLM predictions which are used by the PSL model
 - generate_processed_data contains a script for transforming all of the data into the format that PSL can use 

## Model 
- PSL code
    - training - trains a PSL model and uses the model to make inferences 
    - inference - generates the results discussed in the paper  

## Experiments 
- the code to reproduce results in paper

## Evaluation and Analysis
- a notebook for generating topic assignments
- a folder with prompts
- a script for reading from results and generating metrics 

## Citation
```
@inproceedings{sustainability_signals,
  title     = {{PUBLICSPEAK: Hearing the Public with a Probabilistic Framework}},
  author    = {Xu, Tianliang and Brown, Eva Maxfield and Dwyer, Dustin and Tomkins, Sabina},
  booktitle = {Proceedings of The 39th Annual AAAI Conference on Artificial Intelligence},
  year      = {2025},
  note      = {AI for Social Impact Track},
}
```
