from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.chunking import Chunk, chunk_corpus
from src.embeddings import GeminiEmbedder
from src.generation import GeminiGenerator
from src.ingest import load_corpus 


@dataclass
class SearchResult: 
    chunk: Chunk
    score: float

def normalize_rows(vectors: np.ndarray) -> np.ndarray: 
    norms = np.linalg.norm(vectors, axis = 1, keepdims = True)

    safe_norms = np.clip(norms, 1e-12, None)

    return vectors / safe_norms

class InMemoryVectorStore: 
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray) -> None: 
        if len(chunks) != len(embeddings): 
            raise ValueError("Количество chunks не совпадает с количеством embeddings")

        self.chunks = chunks
        self.embeddings = normalize_rows(embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int=5) -> list[SearchResult]: 
        query_norm = np.linalg.norm(query_embedding)

        if query_norm == 0: 
            raise ValueError("Query embedding имеет нулевую длину")

        normalized_query = query_embedding / query_norm 

        scores = np.einsum("ij,j->i", self.embeddings, normalized_query)

        best_indices = np.argsort(scores)[::-1][:top_k]

        return [SearchResult(
            chunk = self.chunks[index], 
            score = float(scores[index]))
            for index in best_indices]

def print_results(results: list[SearchResult]) -> None: 
    for position, result in enumerate(results, start = 1): 
        metadata = result.chunk.metadata

        print()
        print(
            f"{position}. score={result.score:.4f} | "
            f"document={result.chunk.document_id} | "
            f"model={metadata.get('model', '-')} | "
            f"section={metadata['section']} | "
            f"error={metadata.get('error_code', '-')}"
        )

        preview = result.chunk.text.replace("\n", " ")
        print(preview[:250])

if __name__ == "__main__": 
    documents = load_corpus(Path("data/raw"))
    chunks = chunk_corpus(documents)

    embedder = GeminiEmbedder()
    generator = GeminiGenerator() 

    print(f"Создаем embeddings для {len(chunks)} chunks...")

    chunk_embeddings = embedder.embed_documents([chunk.embedding_text for chunk in chunks])

    store = InMemoryVectorStore(chunks = chunks, embeddings = chunk_embeddings)

    print("Индекс готов. Для выхода введи exit.")

    while True:
        try:
            query = input("\nВопрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if query.casefold() == "exit":
            break 

        if not query: 
            continue 

        query_embedding = embedder.embed_query(query)

        results = store.search(query_embedding = query_embedding, top_k = 8)

        print_results(results) # Вывод чанков

        answer = generator.answer(question = query, results = results)

        print(f"\nОтвет RAG: \n{answer}")
