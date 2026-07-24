"""
This module will load the data from the data folder and prepare it to be embedded and stored in the datastore.
"""

import json
import logging
from pathlib import Path
import time
from uuid import uuid4

from pydantic import BaseModel
from tqdm.auto import tqdm

from .components.dataLoader import dataLoadingUtility
from .components.embeddings import SPLADE, BioBERT, Specter2
from .components.embeddingStore import (
    ChromaStore,
    PineconeStore_Dense,
    QdrantStore_Rerank,
)

logger = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).parent.parent.parent
DELETE_POLL_INTERVAL_SECONDS = 1
DELETE_RETRIES = 10
DB_PATH = PROJECT_DIR / "data" / "db.json"
DB_PATH.parent.mkdir(exist_ok=True)
MAX_DOCUMENTS = 20  # None = all


def prepareJSON(dataPath: Path, outputFile: str = "prepared_data.json"):
    """
    Prepare json file.

    Args:
        dataPath (Path): Path to the data file.
        outputFile (str): Name of the output file.

    Returns:
        Path: Absolute path to the prepared data file.
    """

    logger.info("Prepare data to embed and store...")
    if not dataPath.exists():
        raise FileNotFoundError(f"Data path not found: {dataPath}")
    if not dataPath.is_absolute():
        dataPath = PROJECT_DIR / "data" / dataPath
    if not (dataPath.exists() and dataPath.is_file()):
        raise FileNotFoundError(f"Data path not found: {dataPath}")
    logger.info(f"Loading data from {dataPath}")

    if dataPath.suffix.lower() == ".json":
        logger.info("Data found with .json extension, skipping conversion.")
        return dataPath
    elif dataPath.suffix.lower() != ".ris":
        raise ValueError("Unsupported data format. Use .json or .ris.")

    outputFile = Path(outputFile)
    if not outputFile.is_absolute():
        outputFile = PROJECT_DIR / "data" / outputFile
    if outputFile.exists():
        choice = input(f"Output file {outputFile} already exists. Overwrite? (y/n): ")
        if choice.lower().strip() != "y":
            logger.info(f"Output file {outputFile} already exists. Overwrite? (y/n): ")
            return
        else:
            logger.info(f"Deleting existing file {outputFile}")
            for attempt in range(DELETE_RETRIES):
                try:
                    outputFile.unlink()
                    logger.info(f"Deleted file {outputFile}")
                except OSError as e:
                    logger.error(f"Error deleting file {outputFile}: {e}")
                if outputFile.exists():
                    logger.warning(
                        f"File {outputFile} still exists after {attempt + 1} attempts. Waiting {DELETE_POLL_INTERVAL_SECONDS} seconds before retry..."
                    )
                    time.sleep(DELETE_POLL_INTERVAL_SECONDS)
                else:
                    break
            else:
                raise OSError(f"Failed to delete file {outputFile} after {DELETE_RETRIES} attempts")

    dataLoadingUtility(dataPath, outputFile)
    logger.info(f"Data prepared successfully at {outputFile}")
    return outputFile


class _DB(BaseModel):
    title: str
    abstract: str


def prepareDatabase(dataFile: Path, dbOnly: bool = False):
    """
    Create all the databases from the data file.

    Args:
        dataFile (Path): Absolute path to JSON the data file.
        dbOnly (bool): Whether to only create the database or also embed and store the data.
    """
    logger.info("Starting database preparation...")
    reset = not dbOnly  # If you want generate just the db refs then donot reset the dbs
    # QdrantStore_Rerank = SPLADE | ChromaStore = BioBERT | PineconeStore_Dense = GoogleEmbeddings/Specter2
    embeddingMap = {
        "QdrantStore_Rerank": SPLADE(),
        "ChromaStore": BioBERT(),
        "PineconeStore_Dense": Specter2(),
    }

    qdrantStore_Rerank = QdrantStore_Rerank(
        sparse_embedding_function=embeddingMap["QdrantStore_Rerank"], reset=reset
    )
    chromaStore = ChromaStore(embedding_function=embeddingMap["ChromaStore"], reset=reset)
    pineconeStore_Dense = PineconeStore_Dense(
        embedding_function=embeddingMap["PineconeStore_Dense"], dimension=768, reset=reset
    )

    if dbOnly:
        return qdrantStore_Rerank, chromaStore, pineconeStore_Dense

    with open(dataFile) as f:
        data = json.load(f)

    recordsAdded = 0

    # Reference to original documents
    db = {}
    for record in data:
        if "abstract" not in record or len(record["abstract"]) < 30:
            continue
        if MAX_DOCUMENTS is not None and recordsAdded >= MAX_DOCUMENTS:
            break
        recordsAdded += 1
        doc_id = str(uuid4())
        db[doc_id] = _DB(title=record["title"], abstract=record["abstract"]).model_dump()
    with open(DB_PATH, "w") as f:
        json.dump(db, f)
    logger.info(f"Data preparation completed. Added {recordsAdded} records.")
    logger.info(f"Adding data to {DB_PATH}...")

    # Adding data
    # Qdrant
    q_documents: list[str] = []
    q_metadatas: list[dict] = []
    q_ids: list[str] = []
    for k, v in tqdm(db.items(), desc="Adding data to QdrantStore_Rerank", colour="green"):
        q_documents.append(v["abstract"])
        q_metadatas.append({"doc_id": k})
        q_ids.append(str(uuid4()))

        q_documents.append(v["title"] + " \n " + v["abstract"])
        q_metadatas.append({"doc_id": k})
        q_ids.append(str(uuid4()))
    qdrantStore_Rerank.add_embeddings(documents=q_documents, metadata=q_metadatas, ids=q_ids)

    # ChromaStore
    c_documents: list[str] = []
    c_metadatas: list[dict] = []
    c_ids: list[str] = []
    for k, v in tqdm(db.items(), desc="Adding data to ChromaStore", colour="green"):
        c_documents.append(v["abstract"])
        c_metadatas.append({"doc_id": k})
        c_ids.append(k)
    chromaStore.add_embeddings(documents=c_documents, metadata=c_metadatas, ids=c_ids)

    # PineconeStore_Dense
    p_documents: list[str] = []
    p_metadatas: list[dict] = []
    p_ids: list[str] = []
    for k, v in tqdm(db.items(), desc="Adding data to PineconeStore_Dense", colour="green"):
        p_documents.append(v["abstract"])
        p_metadatas.append({"doc_id": k})
        p_ids.append(str(uuid4()))

        p_documents.append(v["title"])
        p_metadatas.append({"doc_id": k})
        p_ids.append(str(uuid4()))
    pineconeStore_Dense.add_embeddings(documents=p_documents, metadata=p_metadatas, ids=p_ids)


def main():
    prepareDatabase(Path(r"E:\\Projects\\SSNanopore_rag\\data\\papers.json"))


if __name__ == "__main__":
    from ssnanopore_rag.misc.logging_setup import setup_logging

    logger = logging.getLogger(__name__)
    setup_logging()
    logger.info("Starting SSNanopore RAG pipeline...")

    main()
