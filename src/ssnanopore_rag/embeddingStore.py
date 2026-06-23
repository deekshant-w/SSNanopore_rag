from typing import Any
from typing import Mapping
from abc import abstractmethod
from abc import ABCMeta
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
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
logger = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).parent.parent.parent


class EmbeddingStore(metaclass=ABCMeta):
    @abstractmethod
    def add_embeddings(self, ids: list[str], documents: list[str], metadata: list[dict]) -> None:
        pass

    @abstractmethod
    def query(self, query_texts: list[str], n_results: int = 5) -> dict:
        pass


class ChromaStore(EmbeddingStore):
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


class LocalPineconeStore(EmbeddingStore):
    def __init__(self, pinecone_args: Mapping[str, Any] = {}) -> None:
        self._local_host = "http://localhost:5080"
        self.pc = Pinecone(
            api_key="[ENCRYPTION_KEY]",
            host=self._local_host,
            ssl_verify=False,
            **pinecone_args,
        )
        self.init_index()
        index_info = self.pc.indexes.describe(name=self.index_name)
        logger.info(f"Index status: {index_info}")
        data_host = index_info.host.replace("https://", "http://")
        logger.warning(f"Using data host: {data_host}")
        self.index = self.pc.index(name=self.index_name, host=data_host)

    @abstractmethod
    def init_index(self) -> None:
        self.index_name = "<|Uninitialized|>"

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

class PineconeStore_Dense(LocalPineconeStore):
    def __init__(
        self,
        embedding_function: Callable,
        dimension: int,
        index_name: str,
        metric: str = "euclidean",
        namespace: str = "",
        pinecone_args: Mapping[str, Any] = {},
        index_args: Mapping[str, Any] = {},
    ) -> None:
        """
        Pinecone -> index (dense) -> multiple namespaces -> multiple vectors
        """
        self.embedding_function = embedding_function
        self.index_name = index_name
        self.vector_type = "dense"
        self.dimension = dimension
        self.metric = metric
        self.index_args = index_args
        self.namespace = namespace
        super().__init__(**pinecone_args)


    def init_index(self) -> None:
        if not self.pc.has_index(self.index_name):
            logger.info(f"Creating index {self.index_name}")
            self.pc.indexes.create(
                name=self.index_name,
                vector_type="dense",
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                **self.index_args,
            )
        # self.index is defined in the parent class
        
    def add_embeddings(
        self,
        documents: list[str],
        metadata: list[dict],
        ids: list[str],
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
            namespace=self.namespace,
        )
        self.wait_for_upsert(self.index, self.namespace, len(ids))

    def query(self, query_texts: list[str], n_results: int = 5) -> dict:
        query_embeddings = self.embedding_function(query_texts)
        return self.index.query(
            vector=query_embeddings[0], top_k=n_results, namespace=self.namespace
        )


class PineconeStore_Sparse(LocalPineconeStore):
    def __init__(
        self,
        embedding_function: Callable,
        index_name: str,
        metric: str = "dotproduct",
        namespace: str = "",
        pinecone_args: Mapping[str, Any] = {},
        index_args: Mapping[str, Any] = {},
    ) -> None:
        self.embedding_function = embedding_function
        self.index_name = index_name
        self.metric = metric
        self.index_args = index_args
        self.namespace = namespace
        super().__init__(**pinecone_args)


    def init_index(self) -> None:
        if not self.pc.has_index(self.index_name):
            logger.info(f"Creating index {self.index_name}")
            self.pc.indexes.create(
                name=self.index_name,
                vector_type="sparse",
                dimension=1,
                metric=self.metric,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                **self.index_args,
            )
        # self.index is defined in the parent class
        
    def add_embeddings(
        self,
        documents: list[str],
        metadata: list[dict],
        ids: list[str],
    ) -> None:
        embeddings = self.embedding_function(documents)
        logger.info(f"Generated {len(embeddings)} embeddings.")
        self.index.upsert(
            vectors=[
                {
                    "id": ids[i],
                    "sparse_values": embeddings[i],
                    "metadata": metadata[i],
                }
                for i in range(len(embeddings))
            ],
            namespace=self.namespace,
        )
        self.wait_for_upsert(self.index, self.namespace, len(ids))

    def query(self, query_texts: list[str], n_results: int = 5) -> dict:
        query_embeddings = self.embedding_function(query_texts)
        return self.index.query(
            vector=query_embeddings[0], top_k=n_results, namespace=self.namespace
        )



class QdrantStore(EmbeddingStore):
    def __init__(
        self,
        db_name: str = "chroma",
        collection_name: str = "nanopore",
        embedding_function: Optional[Callable] = None,
        vector_size: int = 128,
        metric: str = Distance.COSINE,
    ):
        # client = QdrantClient(path=PROJECT_DIR / "data" / db_name)
        self.client = QdrantClient(":memory:")
        # client = QdrantClient(host="localhost", port=6333)
        self.client.create_collection(
            collection_name,
            vectors_config=VectorParams(size=vector_size, distance=metric)
        )
        self.embedding_function = embedding_function
        self.collection_name = collection_name

    def add_embeddings(
        self,
        documents: list[str],
        metadata: list[dict],
        ids: list[str],
    ) -> None:
        embeddings = self.embedding_function(documents)
        points = [
            PointStruct(
                id=ids[i],
                vector=embeddings[i],
                payload=metadata[i],
            )
            for i in range(len(embeddings))
        ]
        operation_info = self.client.upsert(self.collection_name, points=points, wait=True)
        logger.info(f"Upsert operation completed: {operation_info}")

    def query(self, query_texts: list[str], n_results: int = 5) -> dict:
        query_embeddings = self.embedding_function(query_texts)
        return self.client.query_points(
            collection_name = self.collection_name,
            query = query_embeddings[0],
            limit = n_results,
            with_payload = True,
        )



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


def _pinecone_dense():
    from .embeddings import GoogleEmbeddings as embeddingService

    store = PineconeStore_Dense(
        embedding_function=embeddingService().getEmbeddings, 
        dimension=128,
        index_name="testing"
    )
    store.add_embeddings(
        ids=["good", "bad"],
        documents=["what is DNA sequencing?", "what is CRISPR?"],
        metadata=[{"title": "DNA Sequencing"}, {"title": "CRISPR"}],
    )
    results = store.query(query_texts=["DNA"], n_results=1)
    print(results)


def _pinecone_sparse():
    from .embeddings import SPLADE as embeddingService

    store = PineconeStore_Sparse(
        embedding_function=embeddingService().getEmbeddings,
        index_name="testing",
    )
    store.add_embeddings(
        ids=["good", "bad"],
        documents=["what is DNA sequencing?", "what is CRISPR?"],
        metadata=[{"title": "DNA Sequencing"}, {"title": "CRISPR"}],
    )
    results = store.query(query_texts=["DNA"], n_results=1)
    print(results)


def _qdrant():
    from .embeddings import GoogleEmbeddings as embeddingService

    store = QdrantStore(embedding_function=embeddingService().getEmbeddings, vector_size=128)
    store.add_embeddings(
        ids=[0,1],
        documents=["what is DNA sequencing?", "what is CRISPR?"],
        metadata=[{"title": "DNA Sequencing"}, {"title": "CRISPR"}],
    )
    results = store.query(query_texts=["DNA"], n_results=1)
    print(results)

def main():
    # _chroma()
    # _pinecone_dense()
    # _qdrant()
    _pinecone_sparse()


if __name__ == "__main__":
    main()
