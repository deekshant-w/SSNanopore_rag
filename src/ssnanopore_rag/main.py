import logging
from pathlib import Path
import shutil

import dotenv
import typer

from ssnanopore_rag.misc.logging_setup import setup_logging

app = typer.Typer(help="SSNanopore RAG CLI", no_args_is_help=True)


logger = logging.getLogger(__name__)
dotenv.load_dotenv()
setup_logging()


@app.command()
def prepare(path: str):
    """Prepare a .json or .ris file, then embed and store it. Raises on any other format."""

    from ssnanopore_rag.prepare import prepareDatabase, prepareJSON

    p = Path(path)
    match p.suffix.lower():
        case ".json":
            dataFile = p
        case ".ris":
            dataFile = prepareJSON(p)
        case other:
            raise typer.BadParameter(f"Unsupported format {other!r}: provide a .json or .ris file.")
    prepareDatabase(dataFile)


@app.command()
def run():
    """Load the tools and start the interactive RAG chat loop."""
    from ssnanopore_rag.components.localLLM import LLM, ask_user, welcome
    from ssnanopore_rag.tools import get_tools_and_functions

    welcome()
    tools, functions = get_tools_and_functions()
    llm = LLM(tools=tools, functions=functions)
    while (query := ask_user().strip()) not in ("", "exit", "quit"):
        llm.call(query)


@app.command()
def init():
    """Clear the vector store, then verify that the docker images are reachable."""

    db_path = Path(__file__).parent.parent.parent / "vectorDb"
    shutil.rmtree(db_path, ignore_errors=True)
    db_path.mkdir(exist_ok=True)

    if not _qdrant_up():
        DOCKER_CMD = "docker compose --profile qdrant up"
        typer.echo(f"Containers not reachable. Start them with:\n    {DOCKER_CMD}")

    if not _pinecone_up():
        DOCKER_CMD = "docker compose --profile pinecone up"
        typer.echo(f"Containers not reachable. Start them with:\n    {DOCKER_CMD}")

    typer.echo("Services are reachable. You can start preparing your database.")


def _qdrant_up() -> bool:
    from json import JSONDecodeError

    import requests
    from requests.exceptions import ConnectionError

    url = "http://localhost:6333/collections"
    try:
        result = requests.get(url)
    except ConnectionError as e:
        logger.error(f"Qdrant is not reachable. ConnectionError: {e}")
        return False
    if result.status_code != 200:
        logger.error(f"Qdrant is not reachable. Status code: {result.status_code}")
        return False
    try:
        data = result.json()
    except JSONDecodeError:
        logger.error(f"Qdrant is not reachable. JSONDecodeError: {result.text}")
        return False

    if data.get("status") != "ok":
        logger.error(f"Qdrant is not reachable. Status is not ok: {data}")
        return False

    return True


def _pinecone_up() -> bool:
    from pinecone.errors.exceptions import PineconeConnectionError

    from ssnanopore_rag.components.embeddingStore import PineconeStore_Dense

    try:
        PineconeStore_Dense(embedding_function=lambda _: _, dimension=100).ping()
        return True
    except PineconeConnectionError:
        logger.error("Pinecone is not reachable")
        return False


def main():
    app()


if __name__ == "__main__":
    main()
