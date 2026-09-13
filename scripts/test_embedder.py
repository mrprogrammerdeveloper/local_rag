from local_rag.embeddings.embedder import Embedder


embedder = Embedder()


text = """
Finite element analysis is used
to evaluate mechanical properties.
"""


vector = embedder.embed_text(text)


print(
    "Vector size:",
    len(vector)
)


print(
    vector[:10]
)