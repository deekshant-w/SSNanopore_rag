from google.genai._interactions.types import code_execution_result_step_param
from transformers import AutoTokenizer, AutoModelForMaskedLM
from adapters import AutoAdapterModel
from abc import ABCMeta, abstractmethod
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import logging
import torch

logger = logging.getLogger(__name__)

load_dotenv()


class EmbeddingService(metaclass=ABCMeta):
    def __init__(self):
        self.initializeModelRequirements()

    @abstractmethod
    def initializeModelRequirements(self):
        pass

    @abstractmethod
    def getEmbeddings(self, queries: list[str]) -> list[list[float]]:
        pass


class BioBERT(EmbeddingService):
    def __init__(self):
        logger.info("Initializing BioBERT")
        super().__init__()

    def initializeModelRequirements(self):
        self.tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
        self.model = AutoAdapterModel.from_pretrained("dmis-lab/biobert-v1.1")

    def getEmbeddings(self, queries: list[str]) -> list[list[float]]:
        embeddings = []
        for query in queries:
            inputs = self.tokenizer(
                query,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            outputs = self.model(**inputs)
            embeddings.append(outputs.last_hidden_state[0, 0].detach().numpy().tolist())
        return embeddings


class Specter2(EmbeddingService):
    def __init__(self):
        logger.info("Initializing Specter2")
        super().__init__()

    def initializeModelRequirements(self):
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        self.model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
        self.model.load_adapter(
            "allenai/specter2", source="hf", load_as="specter2", set_active=True
        )
        self.model.set_active_adapters("specter2")
        self.model.eval()

    def getEmbeddings(self, queries: list[str]) -> list[list[float]]:
        embeddings = []
        for query in queries:
            inputs = self.tokenizer(
                query,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            outputs = self.model(**inputs)
            embeddings.append(outputs.last_hidden_state[0, 0].detach().numpy().tolist())
        return embeddings


class GoogleEmbeddings(EmbeddingService):
    def __init__(self):
        logger.info("Initializing GoogleEmbeddings")
        super().__init__()

    def initializeModelRequirements(self):
        self.model = genai.Client()
        self.model_name = "gemini-embedding-2"
        self.output_dimensionality = 128  # None

    def getEmbeddings(self, queries: list[str]) -> list[list[float]]:
        embeddings = []
        for query in queries:
            embeddings.append(
                self.model.models.embed_content(
                    model=self.model_name,
                    contents=query,
                    config=types.EmbedContentConfig(
                        output_dimensionality=self.output_dimensionality
                    ),
                )
                .embeddings[0]
                .values
            )
        return embeddings

class SPLADE(EmbeddingService):
    def __init__(self):
        logger.info("Initializing SPLADE")
        super().__init__()

    def initializeModelRequirements(self, model_id="naver/splade-v3"):
        # model_id = "naver/splade-cocondenser-ensembledistil"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForMaskedLM.from_pretrained(model_id).eval()

    def getEmbeddings(self, queries: list[str]) -> list[list[float]]:
        
        embeddings = []
        for query in queries:
            with torch.no_grad():
                inputs = self.tokenizer(
                    query,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                logits = self.model(**inputs).logits
                vec = torch.max(
                        torch.log1p(torch.relu(logits)) * inputs.attention_mask.unsqueeze(-1),
                        dim=1,
                    ).values.squeeze()
            nz = vec.nonzero().squeeze(-1)
            embeddings.append({"indices": nz.tolist(), "values": vec[nz].tolist()})
        return embeddings

        


def main():
    # embeddingService = GoogleEmbeddings()
    # embeddingService = BioBERT()
    # embeddingService = Specter2()
    embeddingService = SPLADE()
    embeddings = embeddingService.getEmbeddings(
        ["what is DNA sequencing?", "what is CRISPR?"]
    )
    for i, embedding in enumerate(embeddings):
        print(f"Embedding for query {i}: {embedding}")


if __name__ == "__main__":
    main()
