import logging
from pathlib import Path
import shutil
import time

import dotenv
from rich.pretty import pprint
from tqdm.auto import trange
import typer

from ssnanopore_rag.misc.logging_setup import setup_logging

app = typer.Typer(help="Fusion RAG CLI", no_args_is_help=True)


logger = logging.getLogger(__name__)
dotenv.load_dotenv()
setup_logging()


@app.command()
def prepare(
    path: str = typer.Argument(..., help="Path to the .json or .ris input file to process."),
    max_documents: int | None = typer.Argument(
        None, help="Maximum number of documents to process.", show_default=True
    ),
):
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
    prepareDatabase(dataFile, max_documents=max_documents)


@app.command()
def run(model: str = typer.Argument("gemma4", help="LLM model name to use from ollama.")):
    """Load the tools and start the interactive RAG chat loop."""
    from ssnanopore_rag.components.localLLM import LLM, ask_user, welcome
    from ssnanopore_rag.tools import _approxAnswer, get_tools_and_functions

    tools, functions = get_tools_and_functions()
    llm = LLM(model, tools=tools, functions=functions)
    welcome(llm.model)

    # Attatch llm instance to approxAnswer tool
    _approxAnswer.llm = llm
    while (query := ask_user().strip()) != "":
        if query.startswith("/"):
            match query[1:].lower():
                case "debug":
                    pprint(llm.msgs)
                case "tools":
                    pprint(tools)
                case "clear":
                    del llm.msgs[1:]  # Keep the system prompt
                    typer.echo("Conversation cleared.")
                case "quit" | "exit" | "kill":
                    break
                case other:
                    typer.echo(
                        f"Unknown command '/{other}'. Available: "
                        "/debug, /tools, /clear, /quit (or a blank line to exit)."
                    )
            continue
        llm.call(query)


@app.command()
def init():
    """Clear the vector store, then verify that the docker images are reachable."""

    db_path = Path(__file__).parent.parent.parent / "vectorDb"
    logger.info(f"Clearing vector store at {db_path}")
    for _ in trange(20, desc="Attempting to delete..."):
        shutil.rmtree(db_path, ignore_errors=True)
        time.sleep(1)
        if not db_path.exists():
            break
    else:
        typer.echo(f"Failed to delete vector store at {db_path}")
    logger.info(f"Vector store at {db_path} deleted.")
    db_path.mkdir(exist_ok=True)

    qdrantSuccess = True
    pineconeSuccess = True

    if not _qdrant_up():
        qdrantSuccess = False

    if not _pinecone_up():
        pineconeSuccess = False

    if qdrantSuccess and pineconeSuccess:
        typer.echo("Services are reachable. You can start preparing your database.")
    else:
        typer.echo("Some services are not reachable. Please start them and try again.")
        if not qdrantSuccess:
            DOCKER_CMD = "docker compose --profile qdrant up"
            typer.echo(f"Containers not reachable. Start them with:\n    {DOCKER_CMD}")
        if not pineconeSuccess:
            DOCKER_CMD = "docker compose --profile pinecone up"
            typer.echo(f"Containers not reachable. Start them with:\n    {DOCKER_CMD}")
        raise typer.Exit(code=1)


def _qdrant_up() -> bool:
    from json import JSONDecodeError

    import requests

    url = "http://localhost:6333/collections"
    try:
        result = requests.get(url, timeout=10)
    except requests.RequestException as e:
        logger.error(f"Qdrant is not reachable. RequestException: {e}")
        return False
    except Exception as e:
        logger.error(f"Qdrant is not reachable. Exception: {e}")
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
    from ssnanopore_rag.components.embeddingStore import PineconeStore_Dense

    try:
        return PineconeStore_Dense(
            embedding_function=lambda _: _, dimension=100, index_name="testing", reset=True
        ).ping()
    except Exception as e:
        logger.error(f"Pinecone is not reachable. Exception: {e}")
        return False


def main():
    app()


if __name__ == "__main__":
    main()
