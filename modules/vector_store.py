"""Persistent vector store using ChromaDB and Sentence-Transformers embeddings.

Responsibilities:
  * Embed document chunks and user questions with the *same* model so that
    similarity scores are meaningful.
  * Store chunks + metadata persistently on disk (survives app restarts).
  * Detect duplicate documents via a content hash (doc_id) and skip them.
  * Support deliberate re-indexing and per-document deletion.
  * Never wipe the whole database on startup.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import chromadb
from chromadb.api.types import QueryResult

from config import CHROMA_COLLECTION_NAME, CHROMA_DIR, EMBEDDING_MODEL
from modules.pdf_processor import DocumentChunk

# Lazy-loaded embedding model (downloaded on first use).
_embedding_model = None


def get_embedding_model():
    """Return the shared Sentence-Transformers model (loaded once)."""
    global _embedding_model
    if _embedding_model is None:
        # Import here so the rest of the app can be imported without torch.
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of strings into vectors using the configured model."""
    if not texts:
        return []
    model = get_embedding_model()
    # tolist() returns a plain Python list of floats (JSON-serialisable).
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).tolist()


def embed_query(text: str) -> List[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]


# ---------------------------------------------------------------------------
# Vector store wrapper
# ---------------------------------------------------------------------------
class VectorStore:
    """Thin wrapper around a ChromaDB persistent collection."""

    def __init__(
        self,
        persist_dir: str = str(CHROMA_DIR),
        collection_name: str = CHROMA_COLLECTION_NAME,
    ) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        # PersistentClient keeps data on disk across restarts.
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # -- basic info ---------------------------------------------------------
    @property
    def count(self) -> int:
        return self.collection.count()

    def list_documents(self) -> List[Dict]:
        """Return a list of unique documents with chunk counts.

        Each entry: {doc_id, filename, chunk_count}.
        """
        if self.count == 0:
            return []

        # Chroma does not expose a "group by metadata" directly, so we fetch
        # all metadata and aggregate in Python. For a college helpdesk the
        # number of chunks is small enough for this to be fine.
        result = self.collection.get(include=["metadatas"])
        metadatas = result.get("metadatas") or []

        docs: Dict[str, Dict] = {}
        for meta in metadatas:
            if not meta:
                continue
            doc_id = meta.get("doc_id", "unknown")
            entry = docs.setdefault(
                doc_id,
                {"doc_id": doc_id, "filename": meta.get("filename", doc_id), "chunk_count": 0},
            )
            entry["chunk_count"] += 1
        return list(docs.values())

    def has_document(self, doc_id: str) -> bool:
        """Return True if a document with this doc_id is already indexed."""
        if not doc_id:
            return False
        result = self.collection.get(
            where={"doc_id": doc_id},
            include=[],
        )
        return len(result.get("ids") or []) > 0

    # -- indexing -----------------------------------------------------------
    def add_chunks(self, chunks: List[DocumentChunk], reindex: bool = False) -> int:
        """Add a list of document chunks to the store.

        If ``reindex`` is False (default) and the document is already present,
        its chunks are skipped. If ``reindex`` is True, the existing chunks
        for that document are replaced.

        Returns the number of chunks actually added.
        """
        if not chunks:
            return 0

        added = 0
        # Group chunks by doc_id so we can deduplicate / replace per document.
        by_doc: Dict[str, List[DocumentChunk]] = {}
        for c in chunks:
            by_doc.setdefault(c.doc_id, []).append(c)

        for doc_id, doc_chunks in by_doc.items():
            already = self.has_document(doc_id)
            if already and not reindex:
                # Skip duplicate document.
                continue
            if already and reindex:
                # Remove old chunks before re-adding.
                self.delete_document(doc_id)

            texts = [c.text for c in doc_chunks]
            embeddings = embed_texts(texts)
            ids = [
                f"{c.doc_id}_p{c.page_number}_c{c.chunk_index}" for c in doc_chunks
            ]
            metadatas = [
                {
                    "doc_id": c.doc_id,
                    "filename": c.filename,
                    "page_number": c.page_number,
                    "chunk_index": c.chunk_index,
                }
                for c in doc_chunks
            ]

            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            added += len(doc_chunks)

        return added

    # -- retrieval ----------------------------------------------------------
    def query(
        self,
        question: str,
        top_k: int = 5,
    ) -> List[Dict]:
        """Retrieve the most relevant chunks for a question.

        Returns a list of dicts with keys: text, filename, page_number,
        doc_id, distance, score.
        """
        if self.count == 0:
            return []

        query_embedding = embed_query(question)
        k = min(top_k, self.count)
        result: QueryResult = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: List[Dict] = []
        for i, doc_id in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            # Chroma's cosine distance is in [0, 2]; similarity = 1 - distance
            # when embeddings are normalised (which we do).
            distance = distances[i] if i < len(distances) else 1.0
            score = max(0.0, 1.0 - float(distance))
            hits.append(
                {
                    "id": doc_id,
                    "text": documents[i] if i < len(documents) else "",
                    "filename": meta.get("filename", "unknown"),
                    "page_number": meta.get("page_number", 0),
                    "doc_id": meta.get("doc_id", ""),
                    "score": score,
                }
            )
        return hits

    # -- deletion -----------------------------------------------------------
    def delete_document(self, doc_id: str) -> bool:
        """Delete all chunks belonging to a single document.

        Returns True if any chunks were deleted. Other documents are left
        untouched.
        """
        if not doc_id or not self.has_document(doc_id):
            return False
        self.collection.delete(where={"doc_id": doc_id})
        return True

    def clear(self) -> None:
        """Remove every chunk from the collection (use with care)."""
        if self.count > 0:
            all_ids = self.collection.get(include=[])["ids"]
            if all_ids:
                self.collection.delete(ids=all_ids)
