import chromadb
from chromadb.config import Settings
from typing import Optional, Callable
import logging
from pathlib import Path

# from pinecone import Pinecone
from pinecone import ServerlessSpec
import time
from pinecone.grpc import PineconeGRPC as Pinecone
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).parent.parent.parent


class ChromaStore:
    def __init__(
        self,
        db_name: str = "chroma",
        collection_name: str = "nanopore",
        embedding_function: Optional[Callable] = None,
    ) -> None:
        logger.info("Initializing ChromaStore")
        self.client = chromadb.PersistentClient(path=PROJECT_DIR / "data" / db_name)
        # self.client = chromadb.Client()
        self.embedding_function = embedding_function
        self.collection = self.client.get_or_create_collection(
            collection_name,
            configuration={
                "hnsw": {
                    "space": "cosine",
                    "ef_search": 200,
                    "ef_construction": 200,
                    "max_neighbors": 20,
                }
            },
        )
        logger.info("ChromaStore initialized")

    def add_embeddings(
        self, ids: list[str], documents: list[str], metadata: list[dict]
    ) -> None:
        embeddings = self.embedding_function(documents)
        logger.info(f"Generated {len(embeddings)} embeddings.")
        self.collection.add(
            ids=ids, embeddings=embeddings, metadatas=metadata, documents=documents
        )

    def query(self, query_texts: list[str], n_results: int = 5) -> dict:
        return self.collection.query(
            query_embeddings=self.embedding_function(query_texts), n_results=n_results
        )


class PineconeStore:
    def __init__(
        self,
        embedding_function: Optional[Callable] = None,
        dimension: int = None,
        index_name: str = "testing",
        vector_type: str = "dense",
        metric: str = "cosine",
        pinecone_args: Optional[dict] = {},
        index_args: Optional[dict] = {},
    ) -> None:
        logger.info(f"PineconeStore initialized with index {index_name}")
        if embedding_function and not dimension:
            raise ValueError(
                "Dimension must be specified if embedding function is provided"
            )
        self.index_name = index_name
        self.embedding_function = embedding_function
        self._local_host = "http://localhost:5080"
        self.pc = Pinecone(
            api_key="[ENCRYPTION_KEY]",
            host=self._local_host,
            ssl_verify=False,
            **pinecone_args,
        )
        if not self.pc.has_index(index_name):
            logger.info(f"Creating index {index_name}")
            self.pc.indexes.create(
                name=index_name,
                vector_type=vector_type,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                **index_args,
            )
        index_info = self.pc.indexes.describe(name=index_name)
        logger.info(f"Index status: {index_info}")
        data_host = index_info.host.replace("https://", "http://")
        logger.warning(f"Using data host: {data_host}")
        self.index = self.pc.index(name=index_name, host=data_host)

    def add_embeddings(
        self,
        documents: list[str],
        metadata: list[dict],
        ids: list[str],
        namespace: str = "testing",
    ) -> None:
        embeddings = self.embedding_function(documents)
        logger.info(f"Generated {len(embeddings)} embeddings.")
        self.index.upsert(
            vectors=[
                {
                    "id": ids[i],
                    "values": embeddings[i],
                    "metadata": metadata[i],
                }
                for i in range(len(embeddings))
            ],
            namespace=namespace,
        )
        self.wait_for_upsert(self.index, namespace, len(ids))

    def query(
        self, query_texts: list[str], n_results: int = 5, namespace: str = "testing"
    ) -> dict:
        query_embeddings = self.embedding_function(query_texts)
        return self.index.query(
            vector=query_embeddings[0], top_k=n_results, namespace=namespace
        )

    def __del__(self):
        self.pc.indexes.delete(name=self.index_name)
        logger.info(f"Deleted index {self.index_name}")

    def wait_for_upsert(self, index, namespace, expected_count, timeout=100):
        """Block until the index has indexed all records."""
        start = time.time()
        while time.time() - start < timeout:
            stats = index.describe_index_stats()
            ns_stats = stats.namespaces.get(namespace, None)
            if ns_stats and ns_stats.vector_count >= expected_count:
                return
            time.sleep(0.5)
        raise TimeoutError(
            f"Still only {ns_stats.vector_count if ns_stats else 0} vectors after {timeout}s"
        )


class QdrantStore:
    def __init__(): ...


def _chroma():
    from .embeddings import GoogleEmbeddings as embeddingService

    store = ChromaStore(embedding_function=embeddingService().getEmbeddings)
    store.add_embeddings(
        ids=["good", "bad"],
        documents=["what is DNA sequencing?", "what is CRISPR?"],
        metadata=[{"title": "DNA Sequencing"}, {"title": "CRISPR"}],
    )
    results = store.query(query_texts=["DNA"], n_results=1)
    print(results)


def _pinecone():
    from .embeddings import GoogleEmbeddings as embeddingService

    store = PineconeStore(
        embedding_function=embeddingService().getEmbeddings, dimension=128
    )
    store.add_embeddings(
        ids=["good", "bad"],
        documents=["what is DNA sequencing?", "what is CRISPR?"],
        metadata=[{"title": "DNA Sequencing"}, {"title": "CRISPR"}],
    )
    results = store.query(query_texts=["DNA"], n_results=1)
    print(results)


if __name__ == "__main__":
    main()
