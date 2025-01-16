import os
import subprocess
from typing import Optional, List, Dict, Any
import time
import psutil
import GPUtil
import re
from pytube import YouTube
import matplotlib.pyplot as plt
import whisper
from whisperx import load_align_model, align
from whisperx.diarize import DiarizationPipeline, assign_word_speakers
import json
import sys

def download_youtube_video(url: str, output_path: Optional[str] = None) -> None:
    """
    Downloads a YouTube video from the given URL and saves it to the specified output path or the current directory.
    Args:
        url: The URL of the YouTube video to download.
        output_path: The path where the downloaded video will be saved. If None, the video will be saved to the current
        directory.
    Returns:
        None
    """
    yt = YouTube(url)

    video_stream = (
        yt.streams.filter(progressive=True, file_extension="mp4")
        .order_by("resolution")
        .desc()
        .first()
    )
    
    video_name = video_filename(video_stream.title) + ".mp4"

    if output_path:
        video_stream.download(output_path, filename=video_name)
        print(f"Video successfully downloaded to {output_path}")
    else:
        video_stream.download(filename=video_name)
        print("Video successfully downloaded to the current directory")
        

def convert_to_wav(input_file: str, output_file: Optional[str] = None) -> None:
    """
    Converts an audio file to WAV format using FFmpeg.
    Args:
        input_file: The path of the input audio file to convert.
        output_file: The path of the output WAV file. If None, the output file will be created by replacing the input file
        extension with ".wav".
    Returns:
        None
    """
    if not output_file:
        output_file = os.path.splitext(input_file)[0] + ".wav"

    command = f'ffmpeg -n -i "{input_file}" -vn -acodec pcm_s16le -ar 44100 -ac 1 "{output_file}"'

    try:
        subprocess.run(command, shell=True, check=True)
        print(f'Successfully converted "{input_file}" to "{output_file}"')
    except subprocess.CalledProcessError as e:
        print(f'Error: {e}, could not convert "{input_file}" to "{output_file}"')
        
def video_filename(s: str, max_length: int = 255) -> str:
    """Sanitize a string making it safe to use as a filename.
    Args:
        s: A string to make safe for use as a file name.
        max_length:The maximum filename character length.
    Returns:
        A sanitized string.
    """
    # Characters in range 0-31 (0x00-0x1F) are not allowed in ntfs filenames.
    ntfs_characters = [chr(i) for i in range(0, 31)]
    characters = [
        r'"',
        r"\#",
        r"\$",
        r"\%",
        r"'",
        r"\*",
        r"\,",
        r"\.",
        r"\/",
        r"\:",
        r'"',
        r"\;",
        r"\<",
        r"\>",
        r"\?",
        r"\\",
        r"\^",
        r"\|",
        r"\~",
        r"\\\\",
        r" ",
    ]
    pattern = "|".join(ntfs_characters + characters)
    regex = re.compile(pattern, re.UNICODE)
    filename = regex.sub("_", s)
    return filename[:max_length].rsplit(" ", 0)[0]
        
def write_tsv_result(result, file):
    print("start", "end", "speaker", "text", "is_public_comment", "is_public_hearing", "is_comment_trigger", "is_hearing_trigger", sep="\t", file=file)
    for segment in result:
        print(segment["start"], file=file, end="\t")
        print(segment["end"], file=file, end="\t")
        if "speaker" in segment:
            print(segment["speaker"], file=file, end="\t")
        else:
            print("UNKNOWN", file=file, end="\t")
        print(segment["text"].strip().replace("\t", " "), end="\t", file=file)
        print(0, file=file, end="\t")
        print(0, file=file, end="\t")
        print(0, file=file, end="\t")
        print(0, file=file, flush=True)
        
def assign_unknown_speaker(transcript):
    data = []
    for section in transcript["segments"]:
        try:
            start = section["start"]
            end = section["end"]
            text = section["text"]
            speaker = section["speaker"] if "speaker" in section else "UNKNOWN"
            data.append({"start": start, "end": end, "text": text, "speaker": speaker})
        except:
            print(section)

def clean_text(text):
    def remove_punc(text):
        text = re.sub("[^0-9A-Za-z ]", "" , text)
        return text.strip()

    def lemmatize(text):
        lemma = WordNetLemmatizer()
        tokens = text.split()
        return ' '.join([lemma.lemmatize(t, pos = 'v') for t in tokens])
    
    def remove_stop_words(text):
        tokens = text.split()
        filtered_sentence = [w for w in tokens if not w.lower() in stop_words]
        return " ".join(filtered_sentence)
    
    text = text.strip(' ') # strip whitespaces
    text = text.lower() # lowercase
    text = lemmatize(text) # lemmatizing
    text = remove_punc(text) # remove punctuation and symbols
    text = remove_stop_words(text)
    return text


    
if __name__ == "__main__":
    download_youtube_video(sys.argv[1], sys.argv[2])