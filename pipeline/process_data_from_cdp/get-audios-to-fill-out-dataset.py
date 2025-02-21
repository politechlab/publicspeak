from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from cdp_backend.utils.file_utils import resource_copy
from cdp_data import datasets, CDPInstances
import numpy as np
from dataclasses import dataclass

###############################################################################

DATA_DIR = Path(__file__).parent.parent / "data"
EXISTING_METADATA_PATH = (
    (DATA_DIR / "annotation-ready-dataset-metadata.csv").absolute().resolve()
)
NEW_ANNOTATION_SESSION_AUDIOS_DIR = DATA_DIR / "to-annotate-session-audios"
NEW_METADATA_PATH = (
    (DATA_DIR / "extended-annotation-ready-dataset-metadata.csv").absolute().resolve()
)

###############################################################################


@dataclass
class CouncilDatasetRetrievalArgs:
    council: str
    council_short_name: str
    full_council_names: list[str]
    housing_committee_names: list[str]


COUNCIL_DETAIL_LIST = [
    CouncilDatasetRetrievalArgs(
        council=CDPInstances.Seattle,
        council_short_name="seattle",
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


def _generate_dataset(
    target_n_sessions_per_council: int = 20,
) -> pd.DataFrame:
    # Get the full CDP dataset for Seattle, Oakland, and Richmond
    council_datasets = []

    # Iter councils
    for council_and_sample in tqdm(
        COUNCIL_DETAIL_LIST,
        desc="Generating datasets",
    ):
        # Set randomness
        np.random.seed(60)

        # Get all transcripts from the council
        ds = datasets.get_session_dataset(
            council_and_sample.council,
            start_datetime="2021-01-01",
            end_datetime="2022-04-01",
            raise_on_error=False,
        )

        # If council is richmond, drop certain sessions
        if council_and_sample.council_short_name == "richmond":
            remove_sessions = [
                "bbf6a25dae6a",
                "ccfdc9afc698",
                "fe371eafcbd4",
                "528f625abd6a",
                "28db9b12fece",
                "4a32e9a69c09",
                "d4a27b07f47d",
                "d542e7abb073",
                "2462ed64efe2",
                "d8953dadf474",
                "d5e5ef80ef55",
                "a3b5df4a14e3",
                "428c289d4548",
                "612301028fcc",
                "14bce7109743",
                "aa261e4e1e1b",
                "d4ebd22f28e1",
                "be5eeb036b86",
                "e83d3fcee3fa",
                "79a579e7a9ae",
                "02fded472e99",
                "b485034ed953",
                "b1f06d77c1d5",
                "b35666eac4ad",
                "25edf68f4c3e",
                "d1fab39dc2f8",
                "0b604556a687",
            ]

            ds = ds[~ds["id"].isin(remove_sessions)]

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

        # Finally store the audio url
        ds["audio_url"] = ds["session_content_hash"].apply(
            lambda content_hash: f"gs://{council_and_sample.council}.appspot.com/{content_hash}-audio.wav"
        )

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
                "audio_url",
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
        council_datasets.append(ds_metadata)

    # Concatenate all the datasets
    df = pd.concat(council_datasets)

    # Read in the existing metadata file
    existing_annotations = pd.read_csv(EXISTING_METADATA_PATH)

    # Get value counts of existing annotations
    existing_annotations_vc = existing_annotations["council"].value_counts()

    # Find the difference between the target number of sessions and the existing number of sessions
    target_n_sessions_diff = target_n_sessions_per_council - existing_annotations_vc

    # Drop the existing data from the full dataset
    print(f"Number of full council sessions (including existing): {len(df)}")
    df = df[~df["session_id"].isin(existing_annotations["session_id"])]
    print(f"Number of full council sessions (excluding existing): {len(df)}")

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

    # Select rows to fill out the dataset with the target number of sessions
    sampled_council_dfs = []
    for council, n_sessions in target_n_sessions_diff.loc[["richmond"]].items():
        # Get random sample of df for the council
        council_df = df.loc[df["council"] == council].sample(
            n_sessions, random_state=360
        )
        sampled_council_dfs.append(council_df)

    return pd.concat(sampled_council_dfs, ignore_index=True).reset_index(drop=True)


def main() -> None:
    # Prep the dataset
    df = _generate_dataset()

    # Sort by council and then session datetime
    df = df.sort_values(
        ["council", "session_datetime"],
    )

    # Save to disk
    df.to_csv(NEW_METADATA_PATH, index=False)

    # Copy the audios and rename them to the council and session id
    NEW_ANNOTATION_SESSION_AUDIOS_DIR.mkdir(exist_ok=True)
    for _, row in tqdm(
        df.iterrows(),
        desc="Copying audios",
        total=len(df),
    ):
        copy_path = (
            NEW_ANNOTATION_SESSION_AUDIOS_DIR / f"{row.council}-{row.session_id}.wav"
        )
        resource_copy(row.audio_url, copy_path)


if __name__ == "__main__":
    load_dotenv()
    main()
