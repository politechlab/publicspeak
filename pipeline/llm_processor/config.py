"""
Configuration file for LLM processor
"""

# Keywords for filtering content
KEYWORDS = [
    "citizen", "resident", "audience", "crowd", 
    "citizens", "residents", "audiences", 
    "communities", "comment", "comments", 
    "hearing", "hearings",
    "public comment",
    "public hearing",
    "public meeting",
    "public input",
    "public testimony",
    "public forum",
    "public participation",
    "public discussion",
    "public feedback",
    "public opinion"
]

# Prompt templates for different modes
PROMPTS = {
    "find_public_trigger": '''
As a GPT-based meeting transcription post-processor, your task is to identify the start and end of "Public Comments" and "Public Hearings" segments in a meeting transcript. These segments are typically introduced by the meeting manager.

"Public Comments" allow the public to discuss non-agenda items or general government-related matters, while "Public Hearings" are for comments on specific agenda items or legislative matters. Be aware that some meetings may not include these sections, or they might have multiple instances.

To detect the start of these segments, look for cues from the manager such as "start," "public comment on government-related matter," "public comment on legislative matter," "bring us to," and "the first speaker is." For identifying the end, focus on phrases indicating the conclusion of a section, such as "close," "end," and "conclude." Ensure that "comments by council members" do not count as public comments.

Approach the task methodically, and once identified, return the relevant utterances, each with a unique number you assign.

Your output should be in a clean, readable JSON format, strictly adhering to this structure without any extraneous text.

{"public_comments_1":
{"start": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."},
"end": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."}},
"public_comments_2":
{"start": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."},
"end": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."}},
...
"public_hearings_1":
{"start": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."},
"end": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."}, 
"public_hearings_2":
{"start": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."},
"end": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."}}, 
"public_hearings_3":
{"start": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."},
"end": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."}, 
...
}}}
''',

    "find_public_trigger_general": '''
Your role as a GPT-based meeting transcription post-processor is to identify the start and end of "Public Comments" and "Public Hearings" segments in a meeting. "Public Comments" typically but not always, allow the public to discuss non-agenda items, while "Public Hearings" are for comments on specific agenda items. Be aware that some meetings may not include these sections or might have multiple instances.

To detect the start of these segments, look for somebody's call for comments. For identifying the end, focus on phrases indicating the conclusion of a section, such as "bring us to", "close", "start", "end", and "conclude". Approach the task methodically, and once identified, return the relevant utterances, each with a unique number you assign.

{"public_comments_1":
{"start": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."},
"end": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."}},
"public_comments_2":
{"start": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."},
"end": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."}},
...
"public_hearings_1":
{"start": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."},
"end": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."}, 
"public_hearings_2":
{"start": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."},
"end": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."}}, 
"public_hearings_3":
{"start": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."},
"end": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."}, 
...
}}}

You will receive a JSON file containing previously tagged segments of "Public Comments" and "Public Hearings." You should continue adding to the JSON file and return the updated one.

Your output should be in a clean, readable JSON format, strictly adhering to this structure without any extraneous text.
'''
}

# Model configurations
MODEL_CONFIG = {
    "default_model": "gpt-4",
    "temperature": 0,
    "fallback_temperatures": [0.5, 1.0],
    "chunk_size": 3500  # Maximum number of words per chunk
}

# Hallucination indicators
HALLUCINATION_INDICATORS = [
    "https", "http", ".com", ".ai", ".org", ".net", "algorithm", 
    "neural", "whisper", "openai", "machine learning", "training data", "dataset",
    "transcription", "accuracy", "deep learning", "tensor", "compute", "gpu", 
    "castingwords"
] 