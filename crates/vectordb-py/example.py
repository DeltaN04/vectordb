# Example: load sentence-transformer embeddings and run search (requires sentence-transformers)
# pip install sentence-transformers
from sentence_transformers import SentenceTransformer
import vectordb, tempfile, os

model = SentenceTransformer("all-MiniLM-L6-v2")
sentences = ["The cat sits on the mat", "Dogs are loyal", "I love pizza"]
embs = model.encode(sentences).tolist()

d = tempfile.mkdtemp()
c = vectordb.Collection.create(d, "docs", len(embs[0]), "cosine")
for i, e in enumerate(embs):
    c.upsert(i, e, None)
res = c.search(embs[0], 2)
print(res)
EOF