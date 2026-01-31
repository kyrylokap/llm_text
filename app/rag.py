import os
import faiss
import pandas as pd

from sentence_transformers import SentenceTransformer
from .app_logging import logger
from .medline_data_rag import RAG_FILE, download_and_process

_rag_instance = None

def get_rag_service():
    global _rag_instance
    if _rag_instance is None:
        logger.info("[INFO] INITIALIZED RAG")
        _rag_instance = RAG()

        download_and_process()
        _rag_instance.load_medline_csv(RAG_FILE)

        logger.info("[INFO] LOADED RAG CSV")

    return _rag_instance



_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        logger.info("[INFO] Loading Embedding Model (all-MiniLM-L6-v2)...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model



class RAG:
    def __init__(self):
        self.dim = 384
        self.index = faiss.IndexFlatL2(self.dim)
        self.docs: list[dict] = []
        self.model = get_embedding_model()

    def load_medline_csv(self, path: str):
        df = pd.read_csv(path)
        df = df.fillna("")

        texts_to_embed = []
        new_docs = []

        for i , row  in df.iterrows():
            embed_text = (
                f"{row['title']} "
                f"{row['description'][:500]}"
            )

            display_text = (
                f"Disease/Topic: {row['title']}\n"
                f"Description: {row['description']}\n"
            )

            doc_entry = {
                "original_id": row['id'],
                "source": row['source_url'],
                "text": display_text
            }

            new_docs.append(doc_entry)
            texts_to_embed.append(embed_text)

        self.add_docs(texts_to_embed, new_docs)

    def add_docs(self, texts: list[str], doc_objects: list[dict]):
        if not texts:
            return

        vecs = self.model.encode(texts, convert_to_numpy=True)
        self.index.add(vecs)
        self.docs.extend(doc_objects)

    def query(self, text: str, k: int ) -> list[dict]:
        if self.index.ntotal == 0:
            return []

        q_vec = self.model.encode([text], convert_to_numpy=True)
        actual_k = min(k, self.index.ntotal)
        _, idx = self.index.search(q_vec, actual_k)

        return [self.docs[i] for i in idx[0] if i < len(self.docs) and i != -1]
