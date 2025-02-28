Here you can reproduce the results in the paper produced by the learned PublicSpeak model for the city of Ann Arbor. 

Additionally in generate_topic_viz.ipynb notebook, you can run the code to generate the plots exploring the topics that people brought up in their public comments. 

`annotate-topics.py` is a script which uses Anthropic's Claude to annotate individual public comments with a topic classification given some topical seeds. The only added dependency you need is the `anthropic` Python library (`pip install anthropic`). In addition you will need to set the `ANTHROPIC_API_KEY` environment variable to your API key.

To run: `python annotate-topics.py`