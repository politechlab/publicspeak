import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import date

import pandas as pd
import anthropic
from dotenv import load_dotenv
from tqdm import tqdm

###############################################################################

CITY_SHORTCODE_NAME_LUT = {
    "AA": "Ann Arbor, MI",
    "RO": "Royal Oak, MI",
    "JS": "Jackson, MI",
    "LS": "Lansing, MI",
    "RCH": "Richmond, VA",
    "SEA": "Seattle, WA",
    "OAK": "Oakland, CA",
}

CITIES_OF_INTEREST = [
    "AA",
    "RO",
    "JS",
    "LS",
    "SEA",
    "OAK",
    "RCH",
]

# Order data by city population and date
CITY_ORDER = [
    "Seattle, WA",  # 737,015
    "Oakland, CA",  # 440,646
    "Richmond, VA",  # ...
    "Ann Arbor, MI",  # 123,851
    "Lansing, MI",  # 112,644
    "Royal Oak, MI",  # 58,211
    "Jackson, MI",  # 31,309
]

TOPIC_SEEDS = {
    "Housing": [
        "zoning",
        "construction",
        "redevelopment",
        "growth",
        "planning",
        "housing",
        "rent",
        "single family",
        "duplex",
        "apartment",
        "subdivision",
        "renting",
        "rental",
        "landlord",
        "tenant",
        "property",
    ],
    "Transportation and Mobility": [
        "public transit",
        "traffic",
        "bus",
        "car",
        "bike lanes",
        "pedestrian",
        "parking",
        "crosswalk",
    ],
    "Public Safety and Law Enforcement": [
        "police",
        "crime",
        "emergency",
        "safety",
        "property",
        "theft",
        "violence",
        "gun",
        "PD",
    ],
    "Environment and Sustainability": [
        "climate",
        "green",
        "conservation",
        "energy",
        "solar",
        "carbon",
        "pollinators",
        "mow",
        "flood",
        "drought",
        "fire",
    ],
    "Homelessness": [
        "homeless",
        "eviction",
        "shelter",
        "outreach",
        "mental health",
        "substance abuse",
        "housing",
    ],
    "Parks and Recreation": [
        "parks",
        "outdoors",
        "community",
        "events",
        "greenspace",
        "tree",
        "playground",
    ],
    "Economic Development": [
        "business",
        "jobs",
        "tax",
        "revitalization",
        "store",
        "main street",
        "shops",
        "local",
        "cannabis",
    ],
    "Arts and Culture": [
        "events",
        "festivals",
        "museums",
        "performances",
        "sculpture",
        "public art",
        "mural",
        "art",
    ],
    "Education and Youth Services": [
        "schools",
        "libraries",
        "programs",
        "youth",
        "kids",
        "students",
        "teaching",
        "training",
        "games",
        "sports",
    ],
    "Governance and Civic Engagement": [
        "transparency",
        "public participation",
        "elections",
        "accountability",
        "mayor",
        "council",
    ],
    "Israel-Palestine": [
        "Israel",
        "Palestine",
        "genocide",
        "Hamas",
        "Jewish",
        "Muslim",
        "discrimination",
        "Gaza",
        "ceasefire",
    ],
    "Police Reform": [
        "accountability",
        "community oversight",
        "training",
        "defund",
        "reform",
        "police",
        "traffic stops",
        "cops",
        "law",
    ],
    "Utilities": [
        "water",
        "electricity",
        "sewage",
        "internet",
        "utilities",
        "services",
        "DTE",
        "waste",
        "outage",
        "disruption",
        "trees",
        "storm",
        "rates",
        "shutoffs",
        "recycling",
    ],
    "General Community Organizing": [
        "community",
        "services",
        "access",
        "better",
        "organizing",
        "events",
        "accountability ",
        "accountable",
        "help",
        "youth",
        "organization",
        "funding",
        "funds",
        "protect",
        "preserve",
        "group ",
        "petition",
    ],
    "Urban Development": [
        "beautification",
        "historic projects",
        "district",
        "area",
        "history",
        "preservation",
        "development",
        "coliseum",
        "scenic",
        "holiday",
        "lights",
        "tourist",
    ],
}

SYSTEM_MESSAGE_PROMPT = """
You are assisting a computational social science researcher by classifying public comments from city council meetings into various topics. The individual comment to be classified, and the possible topical classifications will be provided to you. Always format your response in XML.
""".strip()  # noqa: E501

USER_MESSAGE_PROMPT = """
### Context

You are tasked with classifying a public comment into a single topic from a provided list.

### Steps

1. First, review the list of topics.

2. Next, carefully read the public comment.

3. Classify this comment into exactly one of the topics from the provided list. Consider the main theme or subject of the comment and how it aligns with the available topics.

4. Before making your final classification, provide your reasoning in two sentences. Explain why you believe the comment fits best into the topic you've chosen. Include this reasoning within <reasoning> tags.

5. After providing your reasoning, state your final classification. Choose only one topic from the list provided, or, if none of the topics are appropriate, choose "Other".

Remember:
- Choose only one topic for classification.
- Provide clear reasoning for your choice.
- Ensure your classification is based solely on the content of the public comment and the provided list of topics, or, if necessary, choose "Other".

Here is an example response structure:

<classification-response>
    <reasoning>...</reasoning>
    <topic>...</topic>
</classification-response>

### Topic Information

{topic_list_str}

### Public Comment

{public_comment}

### Classification

""".strip()  # noqa: E501

TOPIC_SEED_STR = """
{topic_name}
- {keywords}
""".strip()

topic_seed_strs = []
for topic_name, keywords in TOPIC_SEEDS.items():
    topic_seed_strs.append(
        TOPIC_SEED_STR.format(
            topic_name=topic_name,
            keywords="\n- ".join(keywords),
        )
    )

TOPIC_SEED_LIST_STR = "\n\n".join(topic_seed_strs)

ANNOTATIONS_DIR = Path("data/annotated-for-modeling/").resolve()
OUTPUT_FILE = Path("data/full-comment-data-with-topics.csv").resolve()

###############################################################################


def split_short_name_to_city_and_date(short_name: str) -> tuple[str, date]:
    # Split the short name into city and date
    short_code_and_date_parts = short_name.split("_")

    # Short code is the first part
    short_code = short_code_and_date_parts[0]

    # Date is the rest in month day two-digit-year format
    event_date = date(
        year=int("20" + short_code_and_date_parts[-1]),
        month=int(short_code_and_date_parts[1]),
        day=int(short_code_and_date_parts[2]),
    )

    return short_code, event_date


def _prep_dataset() -> pd.DataFrame:
    # Store all data to single object
    data_dfs = []

    # Read all data
    for filepath in ANNOTATIONS_DIR.glob("*.csv"):
        # Read the comment data
        df = pd.read_csv(filepath)

        # Lowercase all columns
        df.columns = df.columns.str.lower()

        # Remove any spaces from column names and replace with "_"
        df.columns = df.columns.str.replace(" ", "_")

        # Split the "name" column into "city_short_code" and "date"
        df["city_short_code"], df["date"] = zip(
            *df["name"].apply(split_short_name_to_city_and_date),
            strict=True,
        )

        # Add the city name
        df["city_name"] = df["city_short_code"].map(CITY_SHORTCODE_NAME_LUT)

        # Add a year-month column
        df["year_month"] = df["date"].apply(lambda x: x.replace(day=1))

        # Using the filename, mark if this was a "training" or "inferred" dataset
        df["dataset_portion"] = filepath.stem.split("_")[-1]

        # Add the truth data to the list
        data_dfs.append(df)

    # Concatenate all training data
    full_data = pd.concat(data_dfs)

    # Replace dataset portion with standard names
    full_data["dataset_portion"] = full_data["dataset_portion"].replace(
        {"truth": "Training", "pred": "Inferred", "val": "Validation"}
    )

    # Subset the data to only the columns we care about
    full_data = full_data[
        [
            "name",
            "city_short_code",
            "city_name",
            "date",
            "year_month",
            "dataset_portion",
            "meeting_section",
            "speaker_role",
            "start",
            "end",
            "text",
        ]
    ]

    # Filter to only the cities of interest
    full_data = full_data[full_data["city_short_code"].isin(CITIES_OF_INTEREST)]

    # Sort by date
    full_data = full_data.sort_values("date", ascending=True)

    # Group by city and then restack
    ordered_city_groups = []
    for city in CITY_ORDER:
        city_data = full_data[full_data["city_name"] == city]
        ordered_city_groups.append(city_data)

    full_data = pd.concat(ordered_city_groups)

    return full_data


def annotate_comments() -> None:
    # Load env
    load_dotenv()

    # Check for multiple keys
    if "ANTHROPIC_API_KEY" in os.environ:
        api_keys = [os.environ["ANTHROPIC_API_KEY"]]
    elif "ANTHROPIC_API_KEY_1" in os.environ:
        api_keys = []
        for env_var in os.environ:
            if env_var.startswith("ANTHROPIC_API_KEY_"):
                api_keys.append(os.environ[env_var])
    else:
        raise ValueError("No API key(s) found in environment variables.")

    # Read and prep data
    full_data = _prep_dataset()

    # Filter out government comments and only use public comment not hearing
    meeting_comments = full_data[
        (full_data["meeting_section"] == "Public Comment")
        & (full_data["speaker_role"] == "Commenter")
    ]

    # Classify all comments
    classified_comments = []
    selected_api_key_index = 0
    for i, row in tqdm(
        meeting_comments.iterrows(),
        desc="Classifying Comments",
        total=len(meeting_comments),
    ):
        # Sleep to avoid rate limiting
        time.sleep(0.5)

        # Get the selected API key
        selected_api_key = api_keys[selected_api_key_index]

        # Increment the index
        selected_api_key_index = selected_api_key_index + 1
        if selected_api_key_index >= len(api_keys):
            selected_api_key_index = 0

        # New client
        client = anthropic.Anthropic(
            api_key=selected_api_key,
        )

        try:
            # Send the message
            message = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                temperature=0,
                system=SYSTEM_MESSAGE_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": USER_MESSAGE_PROMPT.format(
                                    topic_list_str=TOPIC_SEED_LIST_STR,
                                    public_comment=row["text"],
                                ),
                            }
                        ],
                    }
                ],
            )

            # Unpack the content
            content = message.content[0].text

            # Parse the XML
            root = ET.fromstring(content)

            # Get the topic, throw away the reasoning
            topic_el = root.find("topic")
            assert topic_el is not None and topic_el.text is not None

            # Add to the row
            classified_comments.append(
                {
                    **row.to_dict(),
                    "topic": topic_el.text,
                }
            )

        except Exception as e:
            print(f"Exception occurred: {e}")

        # Save every 10 comments
        if i % 10 == 0:
            classified_df = pd.DataFrame(classified_comments)
            classified_df.to_csv(OUTPUT_FILE, index=False)

    # Save the classified comments
    classified_df = pd.DataFrame(classified_comments)
    classified_df.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    annotate_comments()
