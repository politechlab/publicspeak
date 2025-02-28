# Processing and Gathering Data from Council Data Project

Some of the data used for annotation comes from [Council Data Project](https://councildataproject.org/). Specifically we used data from Council Data Project's Seattle, Oakland, and Richmond datasets. The scripts in this directory pull down the audio files and metadata from the CDP API to be passed through the rest of the pipeline.

There are three scripts in this directory:
- `generate-dataset.py`: Used to download the basic JSON and CSV formatted transcripts from the CDP databases. Additionally stores all of the metadata for each of the meetings in a single CSV file for easy lookup and filtering. This metadata file was used by annotators to quickly navigate to the correct meeting webpages they were annotating.
- `diarize-selected-set.py`: Used after annotating the data from `generate-dataset.py` to download the video files associated with each of the "good" quality transcripts. These video files were then passed through the rest of the normal processing pipeline like the rest of the non-CDP data.
- `get-audios-to-fill-out-dataset.py`: Used after we began annotating the CDP data generated from `diarize-selected-set.py`. After we had begun annotating the data, we realized that it would be best to get a few more meetings from each of these city councils. This script downloads only the audio files for the added meetings as we realized we could shorten the processing time be skipping the video to audio conversion step.

To regenerate all of the CDP related data used in this study, each of the scripts should be ran in order from:
1. `python generate-dataset.py`
2. `python diarize-selected-set.py`
3. `python get-audios-to-fill-out-dataset.py`

Note: that by running these scripts, you will be downloading a large amount of video and audio data (> ~10GB). Please ensure you have enough space on your hard drive before running these scripts as well as a stable internet connection.