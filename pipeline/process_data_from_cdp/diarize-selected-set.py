from pathlib import Path
import shutil

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from cdp_backend.utils.file_utils import resource_copy

###############################################################################

DATA_DIR = Path(__file__).parent.parent / "data"
FULL_METADATA_PATH = (DATA_DIR / "full-dataset-metadata.csv").absolute().resolve()
NEW_METADATA_PATH = (
    (DATA_DIR / "annotation-ready-dataset-metadata.csv").absolute().resolve()
)
DIARIZED_TRANSCRIPTS_DIR = DATA_DIR / "diarized-transcripts"

###############################################################################


def _prep_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    # Read in the annotation files
    annotation_details = []
    for short_name in [
        "seattle",
        "oakland",
        "richmond",
    ]:
        council_annotations = pd.read_csv(
            DATA_DIR / f"whole-period-seg-{short_name}.csv"
        )
        council_annotations["council"] = short_name

        # Handle seattle not having a transcript quality column
        if short_name == "seattle":
            council_annotations["transcript_quality"] = "good"

        annotation_details.append(council_annotations)

    # Combine the annotations
    all_annotations = pd.concat(annotation_details)

    # Replace transcript quality values
    all_annotations["transcript_quality"] = all_annotations[
        "transcript_quality"
    ].replace(
        {
            "good-safe-use": "good",
            "good-safe-to-use": "good",
            "okay-use-if-needed": "good",
            "bad-do-not-use": "bad",
        }
    )

    # Groupby council and session_id
    # check period_start_sentence_index or period_end_sentence_index
    # If either is not null, the session has a public comment period
    # Create a row that is council, session_id,
    # has_public_comment_period, and transcript_quality
    def process_council_session_group(group):
        return pd.Series(
            {
                "has_public_comment_period": (
                    group["period_start_sentence_index"].notnull().any()
                    or group["period_end_sentence_index"].notnull().any()
                ),
                "transcript_quality": group["transcript_quality"].iloc[0],
            }
        )

    council_session_annotations = (
        all_annotations.groupby(
            ["council", "session_id"],
        )
        .apply(process_council_session_group)
        .reset_index()
    )

    # Merge the metadata with the annotations
    df = df.merge(council_session_annotations, on=["council", "session_id"])

    # Drop anything with bad transcript quality
    df = df[(df["transcript_quality"] == "good") & (df["has_public_comment_period"])]

    # Convert session_datetime to a datetime object
    df["session_datetime"] = pd.to_datetime(df["session_datetime"])
    df["year"] = df["session_datetime"].dt.year

    # Sort by council and datetime
    df = df.sort_values(["council", "session_datetime"])

    # Subset to full council
    df = df.loc[df["normalized_body_name"] == "full council"]

    # Get 2021 data
    df_2021 = df.loc[df["year"] == 2021].copy()
    print(f"Number of full council sessions (2021): {len(df_2021)}")

    # Get 2022 Jan, Feb, March data
    df_2022 = df.loc[
        (df["year"] == 2022) & (df["session_datetime"].dt.month.isin([1, 2, 3]))
    ].copy()
    print(f"Number of full council sessions (2022 Jan, Feb, March): {len(df_2022)}")

    # Combine the 2021 and 2022 data
    df = pd.concat([df_2021, df_2022])
    return df


def seconds_to_hhmmss(seconds: float) -> str:
    # Remove microseconds
    seconds = int(seconds)
    return str(pd.to_datetime(seconds, unit="s").time())


def main() -> None:
    # Read in the metadata file
    df = pd.read_csv(FULL_METADATA_PATH)

    # Prep the dataset
    df = _prep_dataset(df)

    # Load all annotations
    annotation_dfs = []
    for short_name in ["seattle", "oakland", "richmond"]:
        annotation_df = pd.read_csv(DATA_DIR / f"whole-period-seg-{short_name}.csv")
        annotation_df["council"] = short_name
        annotation_dfs.append(annotation_df)

    # Combine the annotations
    pd.concat(annotation_dfs)

    # Init pipeline
    # pipeline = Pipeline.from_pretrained(
    #     "pyannote/speaker-diarization-3.1",
    #     use_auth_token=os.getenv("HF_AUTH_TOKEN"),
    # )

    # # Try loading pipelines to devices
    # if torch.cuda.is_available():
    #     device = "cuda"
    # # elif torch.backends.mps.is_available():
    # #     device = "mps"
    # else:
    #     device = "cpu"
    #
    # print(f"Using device: {device}")
    # pipeline.to(torch.device(device))

    # Prep out dir
    if DIARIZED_TRANSCRIPTS_DIR.exists():
        shutil.rmtree(DIARIZED_TRANSCRIPTS_DIR)

    DIARIZED_TRANSCRIPTS_DIR.mkdir(exist_ok=True)

    # Apply pretrained pipeline
    new_metadata_rows = []
    for _, session_details in tqdm(df.iterrows(), desc="Sessions", total=len(df)):
        # Load transcript
        transcript = pd.read_csv(session_details["transcript_as_csv_path"])
        transcript = transcript.dropna(subset=["start_time", "end_time", "text"])

        # TODO: handle multiple comment periods
        # TODO: copy each transcript over to drive
        # TODO: add column with timestamped cdp link

        # Find the matching row in the annotations
        # annotation_rows = all_annotations.loc[
        #     (all_annotations["council"] == session_details["council"])
        #     & (all_annotations["session_id"] == session_details["session_id"])
        # ]

        # # Get the period start and end sentence indices
        # period_start_sentence_indices = annotation_rows["period_start_sentence_index"]
        # period_end_sentence_indices = annotation_rows["period_end_sentence_index"]

        # Iter over rows of the transcript and create the new transcript rows
        # following Michigan format
        transcript_annotated_rows = []
        for _, row in transcript.iterrows():
            # # Get the sentence index
            # sentence_index = row["index"]

            # # Check for transitions
            # if sentence_index in period_start_sentence_indices:
            #     transition = "Comments - Into"
            #     in_comment_period = True
            # elif sentence_index in period_end_sentence_indices:
            #     transition = "Comments - Out of"
            #     in_comment_period = False
            # else:
            #     transition = '""'

            # # Check for meeting section
            # if in_comment_period:
            #     meeting_section = "Public Comment"
            # else:
            #     meeting_section = "Other"

            # Convert start and end times from seconds float to
            # HH:MM:SS string
            start_time = seconds_to_hhmmss(row["start_time"])
            end_time = seconds_to_hhmmss(row["end_time"])

            # Append to transcript_annotated_rows
            transcript_annotated_rows.append(
                {
                    "start": start_time,
                    "end": end_time,
                    "text": row["text"],
                    "transition": '""',
                    "meeting-section": "Other",
                    "speaker-role": "Other",
                }
            )

        # Create a DataFrame from speaker_annotated_transcript_rows
        annotated_transcript = pd.DataFrame(transcript_annotated_rows)

        # Save to the same location as the
        # original transcript with "diarized-" prepended
        council = session_details["council"]
        session_id = session_details["session_id"]
        annotated_transcript_path = (
            DIARIZED_TRANSCRIPTS_DIR / f"{council}-session-{session_id}.csv"
        )
        annotated_transcript.to_csv(
            annotated_transcript_path,
            index=False,
        )

        # Append new metadata row
        new_metadata_rows.append(
            {
                "council": session_details["council"],
                "session_id": session_details["session_id"],
                "session_datetime": session_details["session_datetime"],
                "body_name": session_details["body_name"],
                "normalized_body_name": session_details["normalized_body_name"],
                "cdp_url": session_details["cdp_url"],
                "minutes_pdf_url": session_details["minutes_pdf_url"],
                "source_video_url": session_details["source_video_url"],
            }
        )

    # Remove speaker diarization entirely
    # Current issue is that our timestamp generation is different from the
    # diarization timestamps -- the diarization timestamps seem to be correct,
    # our transcript timestamps seem to be a bit early (but its inconsistent)
    # Instead, we can just use the normal transcript
    # but we wont have a speaker column...

    # Store new metadata file
    new_metadata_df = pd.DataFrame(new_metadata_rows)

    # Sort by council and then session datetime
    new_metadata_df = new_metadata_df.sort_values(
        ["council", "session_datetime"],
    )

    # Save to disk
    new_metadata_df.to_csv(NEW_METADATA_PATH, index=False)

    # Take a sample of three meetings from each council
    COPIED_SOURCE_VIDEOS = DATA_DIR / "copied-source-videos"
    COPIED_SOURCE_VIDEOS.mkdir(exist_ok=True)
    for _, row in tqdm(
        new_metadata_df.iterrows(),
        desc="Copying Videos",
        total=len(new_metadata_df),
    ):
        copy_path = COPIED_SOURCE_VIDEOS / f"{row.council}-{row.session_id}.mp4"
        resource_copy(row.source_video_url, copy_path)


if __name__ == "__main__":
    load_dotenv()
    main()
