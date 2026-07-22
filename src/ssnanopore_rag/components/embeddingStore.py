from abc import ABCMeta, abstractmethod
from collections.abc import Callable, Mapping
import logging
from pathlib import Path
import shutil
import time
from typing import Any

import chromadb

# from pinecone import Pinecone
from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm.auto import trange

logger = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).parent.parent.parent.parent


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
        reset: bool = True,
        embedding_function: Callable | None = None,
    ) -> None:
        logger.info("Initializing ChromaStore")
        db_path = PROJECT_DIR / "vectorDb" / db_name
        # delete and wait for the directory to be deleted
        if reset and db_path.exists():
            logger.info(f"Deleting vector DB {db_path}")
            shutil.rmtree(db_path)
            while db_path.exists():
                time.sleep(1)
                logger.info(f"Waiting for vector DB {db_path} to be deleted...")
        self.client = chromadb.PersistentClient(path=db_path)
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

    def add_embeddings(self, ids: list[str], documents: list[str], metadata: list[dict]) -> None:
        embeddings = self.embedding_function.getEmbeddings(documents)
        logger.info(f"Generated {len(embeddings)} embeddings.")
        self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadata, documents=documents)

    def query(self, query_texts: list[str], n_results: int = 5) -> dict:
        return self.collection.query(
            query_embeddings=self.embedding_function.getEmbeddings(query_texts), n_results=n_results
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

    # def __del__(self):
    #     self.pc.indexes.delete(name=self.index_name)
    #     logger.info(f"Deleted index {self.index_name}")

    def wait_for_upsert(self, index, namespace, expected_count, timeout=10000):
        """Block until the index has indexed all records."""
        start = time.time()
        while time.time() - start < timeout:
            stats = index.describe_index_stats()
            current_count = (
                stats.get("namespaces", {}).get(namespace, {}).get("vector_count", 0)
                if namespace
                else stats.get("total_vector_count", 0)
            )
            if current_count >= expected_count:
                return
            print(
                f"Vectors upserted: {current_count / expected_count * 100:.2f}% : {current_count}/{expected_count}"
            )
            time.sleep(2)
        raise TimeoutError(f"Still only {current_count} vectors after {timeout}s")


class PineconeStore_Dense(LocalPineconeStore):
    def __init__(
        self,
        embedding_function: Callable,
        dimension: int,
        index_name: str = "nanopore",
        metric: str = "euclidean",
        namespace: str = "",
        pinecone_args: Mapping[str, Any] = {},
        index_args: Mapping[str, Any] = {},
        reset: bool = True,
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
        self.reset = reset
        super().__init__(**pinecone_args)

    def init_index(self) -> None:
        if not self.reset:
            return
        if self.pc.has_index(self.index_name):
            logger.info(f"Deleting index {self.index_name}")
            self.pc.indexes.delete(name=self.index_name)
            while self.pc.has_index(self.index_name):
                time.sleep(1)
                logger.info(f"Waiting for index {self.index_name} to be deleted...")
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
        batch_size: int = 100,
    ) -> None:
        embeddings = self.embedding_function.getEmbeddings(documents)
        logger.info(f"Generated {len(embeddings)} embeddings.")
        for start in trange(0, len(ids), batch_size, desc="Upserting to PineconeStore_Dense"):
            end = min(start + batch_size, len(ids))
            self.index.upsert(
                vectors=[
                    {
                        "id": ids[i],
                        "values": embeddings[i],
                        "metadata": metadata[i],
                    }
                    for i in range(start, end)
                ],
                namespace=self.namespace,
            )
        self.wait_for_upsert(self.index, self.namespace, len(ids))

    def query(self, query_texts: list[str], n_results: int = 5) -> dict:
        query_embeddings = self.embedding_function.getEmbeddings(query_texts)
        return self.index.query(
            vector=query_embeddings[0], top_k=n_results, namespace=self.namespace
        )

    def ping(self):
        try:
            self.index.describe_index_stats()
            return True
        except Exception as e:
            logger.error(f"Pinecone is not reachable: {e}")
            return False


class PineconeStore_Sparse(LocalPineconeStore):
    def __init__(
        self,
        embedding_function: Callable,
        index_name: str,
        metric: str = "dotproduct",
        namespace: str = "",
        pinecone_args: Mapping[str, Any] = {},
        index_args: Mapping[str, Any] = {},
        reset: bool = True,
    ) -> None:
        raise NotImplementedError(
            "Sparse encoding in Pinecone is not working - https://github.com/pinecone-io/python-sdk/issues/679"
        )
        self.embedding_function = embedding_function
        self.index_name = index_name
        self.metric = metric
        self.index_args = index_args
        self.namespace = namespace
        self.reset = reset
        super().__init__(**pinecone_args)

    def init_index(self) -> None:
        if not self.reset:
            return
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
        embeddings = self.embedding_function.getEmbeddings(documents)
        logger.info(f"Generated {len(embeddings)} embeddings.")
        self.index.upsert(
            vectors=[
                {
                    "id": ids[i],
                    "sparse_values": embeddings[i],
                    "metadata": metadata[i],
                }
                for i in trange(len(embeddings))
            ],
            namespace=self.namespace,
        )
        self.wait_for_upsert(self.index, self.namespace, len(ids))

    def query(self, query_texts: list[str], n_results: int = 5) -> dict:
        query_embeddings = self.embedding_function.getEmbeddings(query_texts)
        return self.index.query(
            vector=query_embeddings[0], top_k=n_results, namespace=self.namespace
        )


class LocalQdrantStore(EmbeddingStore):
    def __init__(self, collection_name: str, reset: bool) -> None:
        logger.info("Initializing %s", type(self).__name__)
        # client = QdrantClient(path=PROJECT_DIR / "data" / db_name)
        client = QdrantClient(url="http://localhost:6333")
        # client = QdrantClient(":memory:")
        self.client = client
        self.collection_name = collection_name
        if reset:
            logger.info("Deleting collection %s", self.collection_name)
            self.client.delete_collection(collection_name=self.collection_name)
            self.init_collection()
        logger.info("%s initialized", type(self).__name__)

    @abstractmethod
    def init_collection(self) -> None:
        """Create the Qdrant collection with the right vector configuration."""
        pass


class QdrantStore_Dense(LocalQdrantStore):
    def __init__(
        self,
        embedding_function: Callable,
        collection_name: str = "nanopore",
        vector_size: int = 128,
        metric: str = Distance.COSINE,
        reset: bool = True,
    ) -> None:
        self.embedding_function = embedding_function
        self.vector_size = vector_size
        self.metric = metric
        super().__init__(collection_name=collection_name, reset=reset)

    def init_collection(self) -> None:
        self.client.create_collection(
            self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=self.metric),
        )

    def add_embeddings(
        self,
        documents: list[str],
        metadata: list[dict],
        ids: list[str],
    ) -> None:
        embeddings = self.embedding_function.getEmbeddings(documents)
        points = [
            PointStruct(
                id=ids[i],
                vector=embeddings[i],
                payload=metadata[i],
            )
            for i in trange(len(embeddings))
        ]
        operation_info = self.client.upsert(self.collection_name, points=points, wait=True)
        logger.info(f"Upsert operation completed: {operation_info}")

    def query(self, query_texts: list[str], n_results: int = 5) -> dict:
        query_embeddings = self.embedding_function.getEmbeddings(query_texts)
        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_embeddings[0],
            limit=n_results,
            with_payload=True,
        )


class QdrantStore_Sparse(LocalQdrantStore):
    def __init__(
        self,
        embedding_function: Callable,
        collection_name: str = "nanopore",
        reset: bool = True,
    ) -> None:
        self.embedding_function = embedding_function
        super().__init__(collection_name=collection_name, reset=reset)

    def init_collection(self) -> None:
        self.client.create_collection(
            self.collection_name,
            vectors_config={},
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )

    def add_embeddings(
        self,
        documents: list[str],
        metadata: list[dict],
        ids: list[str],
    ) -> None:
        embeddings = self.embedding_function.getEmbeddings(documents)
        points = [
            PointStruct(
                id=ids[i],
                vector={
                    "sparse": models.SparseVector(
                        indices=embeddings[i]["indices"],
                        values=embeddings[i]["values"],
                    )
                },
                payload=metadata[i],
            )
            for i in trange(len(embeddings))
        ]
        operation_info = self.client.upsert(self.collection_name, points=points, wait=True)
        logger.info(f"Upsert operation completed: {operation_info}")

    def query(self, query_texts: list[str], n_results: int = 5) -> dict:
        query_embedding = self.embedding_function.getEmbeddings(query_texts)[0]
        return self.client.query_points(
            collection_name=self.collection_name,
            query=models.SparseVector(
                indices=query_embedding["indices"],
                values=query_embedding["values"],
            ),
            using="sparse",
            limit=n_results,
            with_payload=True,
        )


class QdrantStore_Hybrid(LocalQdrantStore):
    def __init__(
        self,
        dense_embedding_function: Callable,
        sparse_embedding_function: Callable,
        collection_name: str = "nanopore",
        vector_size: int = 128,
        metric: str = Distance.COSINE,
        reset: bool = True,
    ) -> None:
        self.dense_embedding_function = dense_embedding_function
        self.sparse_embedding_function = sparse_embedding_function
        self.vector_size = vector_size
        self.metric = metric
        super().__init__(collection_name=collection_name, reset=reset)

    def init_collection(self) -> None:
        self.client.create_collection(
            self.collection_name,
            vectors_config={"dense": VectorParams(size=self.vector_size, distance=self.metric)},
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )

    def add_embeddings(
        self,
        documents: list[str],
        metadata: list[dict],
        ids: list[str],
    ) -> None:
        dense_embeddings = self.dense_embedding_function(documents)
        sparse_embeddings = self.sparse_embedding_function.getEmbeddings(documents)
        points = [
            PointStruct(
                id=ids[i],
                vector={
                    "dense": dense_embeddings[i],
                    "sparse": models.SparseVector(
                        indices=sparse_embeddings[i]["indices"],
                        values=sparse_embeddings[i]["values"],
                    ),
                },
                payload=metadata[i],
            )
            for i in trange(len(ids))
        ]
        operation_info = self.client.upsert(self.collection_name, points=points, wait=True)
        logger.info(f"Upsert operation completed: {operation_info}")

    def query(self, query_texts: list[str], n_results: int = 5) -> dict:
        dense_query = self.dense_embedding_function(query_texts)[0]
        sparse_query = self.sparse_embedding_function.getEmbeddings(query_texts)[0]
        return self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(query=dense_query, using="dense", limit=n_results),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_query["indices"],
                        values=sparse_query["values"],
                    ),
                    using="sparse",
                    limit=n_results,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=n_results,
            with_payload=True,
        )


class QdrantStore_BM25(LocalQdrantStore):
    """Full-text lexical search using Qdrant's built-in BM25 sparse model.
    It uses fastembed to compute IDF-weighted BM25 sparse vectors automatically.
    """

    def __init__(
        self,
        collection_name: str = "nanopore",
        model: str = "Qdrant/bm25",
        reset: bool = True,
    ) -> None:
        self.model = model
        super().__init__(collection_name=collection_name, reset=reset)

    def init_collection(self) -> None:
        self.client.create_collection(
            self.collection_name,
            vectors_config={},
            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )

    def add_embeddings(
        self,
        documents: list[str],
        metadata: list[dict],
        ids: list[str],
    ) -> None:
        points = [
            PointStruct(
                id=ids[i],
                vector={"bm25": models.Document(text=documents[i], model=self.model)},
                payload=metadata[i],
            )
            for i in trange(len(documents))
        ]
        operation_info = self.client.upsert(self.collection_name, points=points, wait=True)
        logger.info(f"Upsert operation completed: {operation_info}")

    def query(self, query_texts: list[str], n_results: int = 5) -> dict:
        return self.client.query_points(
            collection_name=self.collection_name,
            query=models.Document(text=query_texts[0], model=self.model),
            using="bm25",
            limit=n_results,
            with_payload=True,
        )


class QdrantStore_Rerank(LocalQdrantStore):
    """
    dense + sparse + BM25 full-text, then ColBERT rerank.
    """

    def __init__(
        self,
        sparse_embedding_function: Callable,
        dense_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_size: int = 384,
        collection_name: str = "nanopore",
        metric: str = Distance.COSINE,
        bm25_model: str = "qdrant/bm25",
        reranker_model: str = "answerdotai/answerai-colbert-small-v1",  # "colbert-ir/colbertv2.0",
        reranker_dim: int = 96,
        prefetch_limit: int = 20,
        reset: bool = True,
    ) -> None:
        self.dense_embedding_model = dense_embedding_model
        self.sparse_embedding_function = sparse_embedding_function
        self.vector_size = vector_size
        self.metric = metric
        self.bm25_model = bm25_model
        self.reranker_model = reranker_model
        self.reranker_dim = reranker_dim
        self.prefetch_limit = prefetch_limit
        super().__init__(collection_name=collection_name, reset=reset)

    def init_collection(self) -> None:
        self.client.create_collection(
            self.collection_name,
            vectors_config={
                "dense": VectorParams(size=self.vector_size, distance=self.metric),
                "reranker": VectorParams(
                    size=self.reranker_dim,
                    distance=self.metric,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                    hnsw_config=models.HnswConfigDiff(m=0),
                ),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(),
                "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF),
            },
        )

    def add_embeddings(
        self,
        documents: list[str],
        metadata: list[dict],
        ids: list[str],
        batch_size: int = 2,
    ) -> None:
        # All at once, makes the system OOM
        sparse_embeddings = self.sparse_embedding_function.getEmbeddings(documents)
        for start in trange(0, len(ids), batch_size, desc="Upserting to QdrantStore_Rerank"):
            end = min(start + batch_size, len(ids))
            points = [
                PointStruct(
                    id=ids[i],
                    vector={
                        "dense": models.Document(
                            text=documents[i], model=self.dense_embedding_model
                        ),
                        "sparse": models.SparseVector(
                            indices=sparse_embeddings[i]["indices"],
                            values=sparse_embeddings[i]["values"],
                        ),
                        "bm25": models.Document(text=documents[i], model=self.bm25_model),
                        "reranker": models.Document(text=documents[i], model=self.reranker_model),
                    },
                    payload=metadata[i],
                )
                for i in range(start, end)
            ]
            operation_info = self.client.upsert(self.collection_name, points=points, wait=True)
        logger.info(f"Upsert operation completed: {operation_info}")

    def query(self, query_texts: list[str], n_results: int = 5) -> dict:
        query_text = query_texts
        sparse_query = self.sparse_embedding_function.getEmbeddings(query_texts)[0]

        return self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=models.Document(text=query_text[0], model=self.dense_embedding_model),
                    using="dense",
                    limit=self.prefetch_limit,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_query["indices"],
                        values=sparse_query["values"],
                    ),
                    using="sparse",
                    limit=self.prefetch_limit,
                ),
                models.Prefetch(
                    query=models.Document(text=query_text[0], model=self.bm25_model),
                    using="bm25",
                    limit=self.prefetch_limit,
                ),
            ],
            query=models.Document(text=query_text[0], model=self.reranker_model),
            using="reranker",
            limit=n_results,
            with_payload=True,
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
        index_name="testing",
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


def _qdrant_dense():
    from .embeddings import GoogleEmbeddings as embeddingService

    store = QdrantStore_Dense(embedding_function=embeddingService().getEmbeddings, vector_size=128)
    store.add_embeddings(
        ids=[0, 1],
        documents=["what is DNA sequencing?", "what is CRISPR?"],
        metadata=[{"title": "DNA Sequencing"}, {"title": "CRISPR"}],
    )
    results = store.query(query_texts=["DNA"], n_results=1)
    print(results)


def _qdrant_sparse():
    from .embeddings import SPLADE as embeddingService

    store = QdrantStore_Sparse(embedding_function=embeddingService().getEmbeddings)
    store.add_embeddings(
        ids=[0, 1],
        documents=["what is DNA sequencing?", "what is CRISPR?"],
        metadata=[{"title": "DNA Sequencing"}, {"title": "CRISPR"}],
    )
    results = store.query(query_texts=["DNA"], n_results=1)
    print(results)


def _qdrant_hybrid():
    from .embeddings import SPLADE, GoogleEmbeddings

    store = QdrantStore_Hybrid(
        dense_embedding_function=GoogleEmbeddings().getEmbeddings,
        sparse_embedding_function=SPLADE().getEmbeddings,
        vector_size=128,
    )
    store.add_embeddings(
        ids=[0, 1],
        documents=["what is DNA sequencing?", "what is CRISPR?"],
        metadata=[{"title": "DNA Sequencing"}, {"title": "CRISPR"}],
    )
    results = store.query(query_texts=["DNA"], n_results=1)
    print(results)


def _qdrant_bm25():
    store = QdrantStore_BM25()
    store.add_embeddings(
        ids=[0, 1],
        documents=["what is DNA sequencing?", "what is CRISPR?"],
        metadata=[{"title": "DNA Sequencing"}, {"title": "CRISPR"}],
    )
    results = store.query(query_texts=["DNA"], n_results=1)
    print(results)


def _qdrant_rerank():
    from .embeddings import SPLADE

    store = QdrantStore_Rerank(
        sparse_embedding_function=SPLADE().getEmbeddings,
    )
    store.add_embeddings(
        ids=[0, 1],
        documents=["what is DNA sequencing?", "what is CRISPR?"],
        metadata=[{"title": "DNA Sequencing"}, {"title": "CRISPR"}],
    )
    results = store.query(query_texts=["DNA"], n_results=1)
    print(results)


def main():
    # _chroma()
    # _pinecone_dense()
    # _qdrant_dense()
    # _qdrant_sparse()
    # _qdrant_hybrid()
    # _qdrant_bm25()
    _qdrant_rerank()


if __name__ == "__main__":
    main()
