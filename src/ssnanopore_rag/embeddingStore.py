import chromadb
from chromadb.config import Settings
from typing import Optional, Callable
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).parent.parent.parent

class ChromaStore:
    def __init__(self, db_name: str = "chroma", collection_name: str = "nanopore", embedding_function: Optional[Callable] = None) -> None:
        logger.info("Initializing ChromaStore")
        # self.client = chromadb.PersistentClient(path=PROJECT_DIR / "data" / db_name)
        self.client = chromadb.Client()
        self.embedding_function = embedding_function
        self.collection = self.client.get_or_create_collection(
            collection_name, 
            configuration={
                "hnsw":{
                    "space":"cosine",
                    "ef_search": 200,
                    "ef_construction":200,
                    "max_neighbors": 20
                }
            }
        )
        logger.info("ChromaStore initialized")

    def add_embeddings(self, ids: list[str], documents: list[str], metadata: list[dict]) -> None:
        embeddings = self.embedding_function(documents)
        logger.info(f"Generated {len(embeddings)} embeddings.")
        self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadata, documents=documents)

    def query(self, query_texts: list[str], n_results: int = 5) -> dict:
        return self.collection.query(query_embeddings=self.embedding_function(query_texts), n_results=n_results)

    
def _chroma():
    from .embeddings import GoogleEmbeddings as embeddingService
    store = ChromaStore(embedding_function=embeddingService().getEmbeddings)
    store.add_embeddings(
        ids=["good", "bad"],
        documents=["what is DNA sequencing?", "what is CRISPR?"],
        metadata=[{"title": "DNA Sequencing"}, {"title": "CRISPR"}]
    )
    results = store.query(query_texts=["DNA"], n_results=1)
    print(results)


if __name__ == "__main__":
    main()