import os

import numpy as np

from yaw.daemon.graph.inference.embedding import PASSAGE_INSTRUCTION, DSPyEmbeddingAdapter, resolve_embedding_profile

KEY = os.environ["UNSLOTH_STUDIO_AUTH_TOKEN"]
profile = resolve_embedding_profile("embeddinggemma-2", "distinct-titled")
adapter = DSPyEmbeddingAdapter(profile, base_url="http://127.0.0.1:8888/v1", api_key=KEY)

docs = [
    "Paris Landmarks\nThe Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
    "Baking\nTo bake sourdough bread, mix flour, water and starter and let it rise overnight.",
    "Containers\nKubernetes schedules containers onto nodes using resource requests and limits.",
]
doc_vecs = adapter.encode(docs)
print("doc matrix:", doc_vecs.shape, "norms:", np.round(np.linalg.norm(doc_vecs, axis=1), 4))
for query, expected in {
    "Where is the Eiffel Tower located?": 0,
    "How do I make sourdough?": 1,
    "How are pods placed on nodes?": 2,
}.items():
    q = adapter.encode([query], instruction=PASSAGE_INSTRUCTION)
    scores = (doc_vecs @ q.T).ravel()
    print(f"{query!r}: best={int(scores.argmax())} expected={expected} scores={np.round(scores, 3)}")
long = adapter.encode(["word " * 3000])
print(
    "3000-word document encoded:", long.shape, "telemetry tail:", adapter.telemetry[-1] if adapter.telemetry else None
)
