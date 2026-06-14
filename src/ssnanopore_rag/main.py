from ssnanopore_rag.dataLoader import (
    Paper,
    parse_ris_data,
    convert_ris_data_to_entities,
    make_data,
)
from ssnanopore_rag.embeddings import (
    EmbeddingService,
    BioBERT,
    Specter2,
    GoogleEmbeddings,
)


def main():
    make_data()

    embeddingService: EmbeddingService = BioBERT()
    embeddings = embeddingService.getEmbeddings(
        ["what is DNA sequencing?", "what is CRISPR?"]
    )
    for i, embedding in enumerate(embeddings):
        print(f"Embedding for query {i}: {embedding}")


if __name__ == "__main__":
    main()
