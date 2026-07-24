from collections import defaultdict
import json
import logging

from sentence_transformers import CrossEncoder

from .components.localLLM import single_turn_llm
from .prepare import DB_PATH, prepareDatabase

logger = logging.getLogger(__name__)


class _approxAnswer:
    template = """
    Given the following query: <|{query}|>

    Answer this query to the best of your knowledge WITHOUT USING ANY TOOLS. This is for approximation only. The answer to this tool will be given to the RAG tool as an input query. So try to answer it the best you can. The only output should be a small paragraph answer to the user's query. Do not display any pretext fluff or post text followups.
    """
    llm = None

    @staticmethod
    def call(query: str) -> str:
        self = _approxAnswer
        text = self.template.format(query=query)
        return single_turn_llm(text, self.llm, True)


approxAnswer = _approxAnswer.call


class _RAG_Tool:
    def __init__(self):
        self.qdrantStore_Rerank, self.chromaStore, self.pineconeStore_Dense = prepareDatabase(
            None, dbOnly=True
        )
        self.count = 100
        self.qCount = self.count
        self.cCount = self.count
        self.pCount = self.count

        self.qWeight = 10
        self.cWeight = 5
        self.pWeight = 2
        self.k = 60

        self.topN = 30  # Combined RRF top results
        self.selectedTopN = 10  # Final top results after reranking
        # self.rerankerName = "BAAI/bge-reranker-v2-m3"
        self.rerankerName = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.reranker = CrossEncoder(self.rerankerName)

    def call(self, query: str):
        qPicks = self.qdrantStore_Rerank.query([query], n_results=self.qCount)
        qDocIds = [i.payload["doc_id"] for i in qPicks.points]

        cPicks = self.chromaStore.query([query], n_results=self.cCount)
        cDocIds = [i["doc_id"] for i in cPicks["metadatas"][0]]

        pPicks = self.pineconeStore_Dense.query([query], n_results=self.pCount)
        pDocIds = [i.metadata["doc_id"] for i in pPicks.matches]

        scores = defaultdict(lambda: 0)
        for i, docId in enumerate(qDocIds, 1):
            scores[docId] += self.qWeight * (1 / (self.k + i))
        for i, docId in enumerate(cDocIds, 1):
            scores[docId] += self.cWeight * (1 / (self.k + i))
        for i, docId in enumerate(pDocIds, 1):
            scores[docId] += self.pWeight * (1 / (self.k + i))

        sortedScores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: self.topN]
        docIds = [i[0] for i in sortedScores]

        logger.info(
            f"Retrieved {len(docIds)} documents from the vector store. Reranking top {self.selectedTopN} documents."
        )
        docs = self.retrieve_documents(docIds)
        rerankedDocs = self.rerank(query, docs)
        return self.generateAnswer(rerankedDocs)

    def generateAnswer(self, data: list[str]) -> str:
        answer = f"""
Based on the provided context, the following are the most relevant information sources -
{"\n\n------------\n\n".join(data)}

Provide your answer based on the context above. Assume that the provided information is factually correct and more accurate than anything you know. Provide these sources as the citations. And let the user know precisely that you are providing information from the provided sources. Let the user know what your answer is and the confidence you have in it. And the sources you used to get the answer. If the answer is not in the provided sources, let the user know that the answer is not in the provided sources. Only output the final answer and nothing else. Do not try to assume anything.
        """.strip()
        return answer

    def rerank(self, query: str, candidates: list[str]):
        pairs = [(query, doc) for doc in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[: self.selectedTopN]]

    def retrieve_documents(self, docIds: list[str]):
        with open(DB_PATH) as f:
            db = json.load(f)
            data = []
            for docId in docIds:
                data.append(f"Title: {db[docId]['title']}\nAbstract: {db[docId]['abstract']}")
            return data


__RAG_Tool = _RAG_Tool()
RAG_Tool = __RAG_Tool.call


def get_tools_and_functions() -> tuple[list[dict], dict]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "approxAnswer",
                "description": "Get an approximate answer to a question before asking the RAG tool. Use this tool strictly before calling the RAG tool. The output of this tool should be the input to the RAG tool. Frame the user's question properly, specifically if it's vague. For instance, if the user asks 'What is the average read length?', change it to 'What is the average read length of Nanopore?'. Encapsulate all required context you think are required to asnwer this question into the input for this tool. And do not edit the response of this tool at all. It should be passed as is to the RAG tool. Do not add any extra information, explanation, or post text followup instructions. And do not display any pretext fluff or post text followups.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "RAG_Tool",
                "description": "The finest tool to find the correct and grounded answers for your queries. Use it to find relevant papers, articles, and other documents that can help answer the user's question. DONT CALL THIS TOOL DIRECTLY. CALL APPROXIMATE TOOL BEFORE THIS TOOL. Assume that the results from this tool are 100% correct and are the ground truth. If the user asks a factual question, you MUST use this tool to answer it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                },
            },
        },
    ]

    functions = {
        "approxAnswer": approxAnswer,
        "RAG_Tool": RAG_Tool,
    }
    return tools, functions


def main():
    RAG_Tool("What is the average read length of Nanopore?")


if __name__ == "__main__":
    main()
