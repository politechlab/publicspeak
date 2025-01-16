import argparse


parser = argparse.ArgumentParser(description='RecPlay')

#########################################################

parser.add_argument('--mode', type=str, default='full')

########################################################## args for general



########################################################## args for video download
parser.add_argument('--url', type=str)
parser.add_argument('--video_output', type=str, default="video_pool")

######################################################### args for video to wav
parser.add_argument('--wav_input', type=str, default='')

######################################################### args for whisperX
parser.add_argument('--audio_file', type=str, default='')
parser.add_argument('--model_name', type=str, default='large-v3')
parser.add_argument('--device', type=str, default='cuda:1,2,3,4,5,6,7')
parser.add_argument('--language_code', type=str, default='en')
parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument('--compute_type', type=str, default='float16')
parser.add_argument('--ts_output_folder', type=str, default="ts_output/new_ts_output")
parser.add_argument('--ts_path', type=str, default="")

######################################################## args for trigger
parser.add_argument('--long_text_th', type=int, default=50)
parser.add_argument('--ratio_count', type=float, default=0.5)
parser.add_argument('--cut_off_th', type=int, default=200)
parser.add_argument('--gpt_version', type=str, default='gpt-4')


########################################################
parser.add_argument('--random_seed', type=int, default=45)


# TODO: Remove the argparse file and set up a .ini file as a configure file.
# TODO: write a README file.
# TODO: Remove the code above the line, and put them into other scripts.  

################
args = parser.parse_args()