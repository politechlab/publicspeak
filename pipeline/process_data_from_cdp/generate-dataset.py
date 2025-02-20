from cdp_data import datasets, CDPInstances
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass

###############################################################################

DATA_DIR = Path(__file__).parent.parent / "data"
FULL_METADATA_PATH = DATA_DIR / "full-dataset-metadata.csv"

###############################################################################


@dataclass
class CouncilDatasetRetrievalArgs:
    council: CDPInstances
    council_short_name: str
    sample: int
    full_council_names: list[str]
    housing_committee_names: list[str]


COUNCILS_AND_SAMPLES = [
    CouncilDatasetRetrievalArgs(
        council=CDPInstances.Seattle,
        council_short_name="seattle",
        sample=200,
        full_council_names=[
            "City Council",
        ],
        housing_committee_names=[
            "Affordable Housing, Neighborhoods, and Finance Committee",
            "Committee on Housing Affordability, Human Services, and Economic Resiliency",  # noqa: E501
            "Finance and Housing Committee",
            "Housing, Health, Energy, and Workers' Rights Committee",
            "Housing, Health, Energy, and Workers’ Rights Committee",
            "Sustainability and Renters' Rights Committee",
            "Select Committee on 2023 Housing Levy",
        ],
    ),
    CouncilDatasetRetrievalArgs(
        council=CDPInstances.Oakland,
        council_short_name="oakland",
        sample=201,
        full_council_names=[
            "Special Concurrent Meeting of the Oakland Redevelopment Successor Agency/City Council",  # noqa: E501
            "Special Concurrent Meeting of the Oakland Redevelopment Successor Agency / City Council / Geologic Hazard Abatement District Board",  # noqa: E501
            "Concurrent Meeting of the Oakland Redevelopment Successor Agency / City Council / Geologic Hazard Abatement District Board",  # noqa: E501
            "* Concurrent Meeting of the Oakland Redevelopment Successor Agency and the City Council",  # noqa: E501
        ],
        housing_committee_names=[
            "*Community & Economic Development Committee",
            "*Special Community & Economic Development Committee",
        ],
    ),
    CouncilDatasetRetrievalArgs(
        council=CDPInstances.Richmond,
        council_short_name="richmond",
        sample=50,
        full_council_names=[
            "City Council",
        ],
        housing_committee_names=[
            "Land Use, Housing and Transportation Standing Committee",
        ],
    ),
]

CDP_URL_TEMPLATE = "https://councildataproject.org/{council_short_name}/#/events/{event_id}?s={session_index}"

###############################################################################


def generate_dataset():
    # Create the overall directory for saving
    storage_dir = Path("transcripts/")
    storage_dir.mkdir(exist_ok=True)

    # Full dataset dataframe list
    full_dataset_list = []

    # Iter councils
    for council_and_sample in tqdm(
        COUNCILS_AND_SAMPLES,
        desc="Generating datasets",
    ):
        # Set randomness
        np.random.seed(60)

        # Get all transcripts from the council
        ds = datasets.get_session_dataset(
            council_and_sample.council,
            store_transcript=True,
            store_transcript_as_csv=True,
            # store_audio=True,
            start_datetime="2020-01-01",
            end_datetime="2024-01-01",
            sample=council_and_sample.sample,
            raise_on_error=False,
        )

        # Create child for this council
        council_dir = storage_dir / council_and_sample.council_short_name
        council_dir.mkdir(exist_ok=True)

        # Iter sessions
        for _, row in ds.iterrows():
            # create the copy path
            transcript_copy_path = council_dir / f"{row['id']}.csv"

            # read the original transcript
            transcript = pd.read_csv(row.transcript_as_csv_path)

            # keep only the index and text columns
            transcript = transcript[
                [
                    "index",
                    "text",
                ]
            ]

            # rename index to sentence_index
            transcript = transcript.rename(columns={"index": "sentence_index"})

            # add column for session id
            transcript["session_id"] = row["id"]

            # add column for council
            transcript["council"] = CDPInstances.Seattle

            # save the modified transcript
            transcript.to_csv(transcript_copy_path, index=False)

        # Add new columns to be stored with the metadata of the dataset
        ds["council"] = council_and_sample.council_short_name
        ds["cdp_url"] = ds.apply(
            lambda x: CDP_URL_TEMPLATE.format(
                council_short_name=council_and_sample.council_short_name,
                event_id=x.event.id,
                session_index=x.session_index,
            ),
            axis=1,
        )
        ds["body_name"] = ds.body.apply(lambda x: x.name)

        def get_committee_type(body_name):
            if body_name in council_and_sample.full_council_names:
                return "full council"
            elif body_name in council_and_sample.housing_committee_names:
                return "housing committee"
            else:
                return "other"

        ds["normalized_body_name"] = ds.body_name.apply(get_committee_type)

        # Subset to only the columns we want to keep
        ds_metadata = ds[
            [
                "council",
                "id",
                "session_datetime",
                "body_name",
                "normalized_body_name",
                "cdp_url",
                "minutes_pdf_url",
                "video_uri",
                "transcript_as_csv_path",
                # "audio_path",
            ]
        ].copy()

        # Rename a few of the columns
        ds_metadata = ds_metadata.rename(
            columns={
                "id": "session_id",
                "video_uri": "source_video_url",
            }
        )

        # Add the council to the full dataset list
        full_dataset_list.append(ds_metadata)

    # Concatenate all the datasets
    full_dataset = pd.concat(full_dataset_list)

    # Save the full dataset
    full_dataset.to_csv(FULL_METADATA_PATH, index=False)


###############################################################################

if __name__ == "__main__":
    generate_dataset()
