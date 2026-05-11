# 🧠 My System: The Aegis Health Graph

I’m building a system that behaves less like a tracker and more like a personal medical reasoning engine.

At a high level, I’m trying to solve this:

People describe their health in messy, incomplete, time-relative language, and I want to convert that into a structured, evolving, queryable memory that can support reasoning.




Current Simplified Issues:

- Temporal metadata and event time in pipeline.py. There are a few if else conditions which define the event time and ultimately the temporal metadata. The user input can vary a lot and most likely wont fit into this simplified framework.

- The symptom normalizer (canonicalizer.py) is basically just converting it into lowercase and has a limited CONCEPT_MAP defined for a handful of symptoms. The user input will most defintely contain tons of other symptoms and the system would have to rely on the LLM to produce normalized output in the first place.

- Same issue as in point 2, the normalize body part uses a similar limited BODY_PART_MAP.

- Find best episode threshold is hardcoded to be > 0.6 episodes.py. And the way the score is calculated is a bit ambiguous, score += 0.5 sim ? how to come up with proper weighted equations. Also TIME_WINDOW_DAYS = 10 ? How to decide that value.

- For node similarities we might wanna use word embedding models instead of these sentence/text embedding models. Coz these always score really high.

- The current retriever system is pretty basic, it just scores based on overlapping tokens. And the retrieval engine is not connected to generator for answering.



FEATS TO ADD:

- Notifications to clarify transient events like cyclic, Do you still go to tennis every week?

- Relevancy calculations when on charge. To update DB. On idle use.


PENDING ITEMS:

- EPISODE CONNECTIONS / MERGE