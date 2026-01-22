import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


class MiniRAG:
    def __init__(self):
        self.dim = 384
        self.index = faiss.IndexFlatL2(self.dim)
        self.docs: list[dict] = []

    def load_csv(
            self,
            path: str,
            text_columns: list[str],
            sep: str = ","
    ):
        df = pd.read_csv(path, sep=sep)

        texts_to_embed = []
        new_docs = []
        start_id = len(self.docs)

        for col in text_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in CSV")

            values = df[col].dropna().astype(str).str.strip().tolist()

            for i, text in enumerate(values):
                doc_entry = {
                    "text": text,
                    "source": path,
                    "chunk_id": start_id + i
                }
                new_docs.append(doc_entry)
                texts_to_embed.append(text)

        self.add_docs(texts_to_embed, new_docs)

    def add_docs(self, texts: list[str], doc_objects: list[dict]):
        if not texts:
            return

        vecs = model.encode(texts, convert_to_numpy=True)
        self.index.add(vecs)
        self.docs.extend(doc_objects)

    def query(self, text: str, k: int = 5) -> list[dict]:
        if self.index.ntotal == 0:
            return []

        q_vec = model.encode([text], convert_to_numpy=True)
        actual_k = min(k, self.index.ntotal)
        _, idx = self.index.search(q_vec, actual_k)

        return [self.docs[i] for i in idx[0] if i < len(self.docs) and i != -1]

    def load_enriched_csv(self, path: str):
        df = pd.read_csv(path)

        required_cols = ["text", "specialist"]
        if not all(col in df.columns for col in required_cols):
            print(f"[WARN] File {path} does not have required columns: {required_cols}")
            return

        texts_to_embed = []
        new_docs = []
        start_id = len(self.docs)

        for i, row in df.iterrows():
            combined_text = (
                f"Description of the disease: {row['text']}\n"
                f"Recommended specialist: {row['specialist']}."
            )

            doc_entry = {
                "text": combined_text,
                "source": path,
                "chunk_id": start_id + i,
            }

            texts_to_embed.append(combined_text)
            new_docs.append(doc_entry)

        self.add_docs(texts_to_embed, new_docs)
