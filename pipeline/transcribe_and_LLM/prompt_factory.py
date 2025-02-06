# initial_prompt = '''
# As a GPT based meeting transcription post-processor, your task is to find out the utterances indicating the start and the end of the section public comments and public hearings. These are the utterance from the meeting manager. Public comment is the section for the public to comments on non-agenda items and public hearings is the section for the public to comments on some particular items on agenda, and your task it to identify the public comments and public hearings but note that some meetings may not have them or have multiple of them. For starting, the manager will call for comments. For ending, you need to first consider the indication of the end of the section. Start identifying key phrases such as "bring us to", "close", "start", "end", "conclude" first. Think of it step by step and return me the entire utterance with the numbering you choose with the numbers. Output: Your output must be strictly in readable JSON format
# without any extra text:
# {"public_comments_1":
# {"start": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."},
# "end": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."}},
# "public_comments_2":
# {"start": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."},
# "end": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."}},
# ...
# "public_hearings_1":
# {"start": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."},
# "end": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."}, 
# "public_hearings_2":
# {"start": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."},
# "end": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."}}, 
# "public_hearings_3":
# {"start": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."},
# "end": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."}, 
# ...
# }}}

# '''

initial_prompt = '''
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

'''

# initial_prompt = '''
# Your role as a GPT-based meeting transcription post-processor is to identify the start and end of "Public Comments" and "Public Hearings" segments in a meeting. These segments are typically introduced by the meeting manager. "Public Comments" allow the public to discuss non-agenda items or government related matters, while "Public Hearings" are for comments on specific agenda items or legislative matters. Be aware that some meetings may not include these sections or might have multiple instances.

# To detect the start of these segments, look for the manager's call for comments, such as "start", "public comment on government related matter", "public comment on legislative matter", "bring us to" and "first speaker is". For identifying the end, focus on phrases indicating the conclusion of a section, such as "close",  "end", and "conclude". Make sure "public comments from council members" does not count for public comments. Approach the task methodically, and once identified, return the relevant utterances, each with a unique number you assign.

# Your output should be in a clean, readable JSON format, strictly adhering to this structure without any extraneous text.

# {"public_comments_1":
# {"start": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."},
# "end": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."}},
# "public_comments_2":
# {"start": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."},
# "end": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."}},
# ...
# "public_hearings_1":
# {"start": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."},
# "end": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."}, 
# "public_hearings_2":
# {"start": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."},
# "end": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."}}, 
# "public_hearings_3":
# {"start": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."},
# "end": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."}, 
# ...
# }}}

# '''

initial_prompt_multi = '''
Your role as a GPT-based meeting transcription post-processor is to identify the start and end of "Public Comments" and "Public Hearings" segments in a meeting. "Public Comments" allow the public to discuss non-agenda items, while "Public Hearings" are for comments on specific agenda items. Be aware that some meetings may not include these sections or might have multiple instances.

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

initial_prompt_general = '''
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

initial_prompt_PC = '''
Your role as a GPT-based meeting transcription post-processor is to identify the start and end of "Public Comments" segments in a meeting. "Public Comments" typically but not always, allow the public to discuss non-agenda items. Be aware that some meetings may not include the section or might have multiple instances.

To detect the start of these segments, look for somebody's call for comments. For identifying the end, focus on phrases indicating the conclusion of a section, such as "bring us to", "close", "start", "end", and "conclude". Approach the task methodically, and once identified, return the relevant utterances, each with a unique number you assign.

{"public_comments_1":
{"start": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."},
"end": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."}},
"public_comments_2":
{"start": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."},
"end": {"numbering": <numbering_of_the_utterance>, "text": <utterance_chosen>, "reasons": "reason1;reason2;..."}},
...
}}}

You will receive a JSON file containing previously tagged segments of "Public Comments". You should continue adding to the JSON file and return the updated one.

Your output should be in a clean, readable JSON format, strictly adhering to this structure without any extraneous text.
'''