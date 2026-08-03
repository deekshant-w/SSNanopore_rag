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
        cls = _approxAnswer
        text = cls.template.format(query=query)
        if not cls.llm:
            raise ValueError("LLM not initialized. Call prepareDatabase first.")
        result = single_turn_llm(text, cls.llm, False)
        result = f"Context Query for RAG: {result} <NOW CALL RAG_TOOL>"
        logger.debug(f"Approximate answer: {result}")
        return result


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
        self.cWeight = 5 * 1.5  # as others have 2 copies of documents
        self.pWeight = 2
        self.k = 60

        self.topN = 30  # Combined RRF top results
        self.selectedTopN = 10  # Final top results after reranking
        # self.rerankerName = "BAAI/bge-reranker-v2-m3"
        self.rerankerName = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.reranker = CrossEncoder(self.rerankerName)

        with open(DB_PATH) as f:
            self.db = json.load(f)

    def call(self, query: str, question: str):
        logger.debug(f"RAG tool called with query: {query}")
        logger.debug(f"RAG tool called with question: {question}")
        qPicks = self.qdrantStore_Rerank.query(
            [query], n_results=self.qCount, individual_limit=int(self.qCount / 1.5)
        )
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
        return self.generateAnswer(rerankedDocs, question)

    def generateAnswer(self, data: list[str], question: str) -> str:
        answer = f"""
Based on the provided context, the following are the most relevant information sources -
{"\n\n------------\n\n".join(data)}

Provide your answer based on the context above. Assume that the provided information is factually correct and more accurate than anything you know. Provide these sources as perfectly placed and useful citations. And let the user know precisely that you are providing information from the provided sources. Let the user know what your answer is and the confidence you have in it. And at the end <Precisely and correcly> always mention the sources you used to get the answer. The user cannot see the internal tool context so always write the sources you used to get the answer clearly with numbers as citations like [1], [2], [3], etc. At the end of the answer draw a horizontal line and add a references section listing all the sources used with those matching numers - eg. [1] title - "title1", [2] title - "title2", etc. If the answer is not in the provided sources, let the user know that the answer is not in the provided sources.

Based on the above context, answer the following question - {question}
        """.strip()
        return answer

    def rerank(self, query: str, candidates: list[str]):
        pairs = [(query, doc) for doc in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[: self.selectedTopN]]

    def retrieve_documents(self, docIds: list[str]):
        data = []
        for docId in docIds:
            data.append(f"Title: {self.db[docId]['title']}\nAbstract: {self.db[docId]['abstract']}")
        return data


__RAG_Tool = _RAG_Tool()
RAG_Tool = __RAG_Tool.call


def get_tools_and_functions() -> tuple[list[dict], dict]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "approxAnswer",
                "description": "Get an approximate answer to a question before asking the RAG tool. Use this tool strictly before calling the RAG tool. The output of this tool should be strictly **DIRECT WORD TO WORD** input to the RAG tool. Frame the user's question properly, specifically if it's vague. For instance, if the user asks 'What is the average read length?', change it to 'What is the average read length of Nanopore?'. Encapsulate all required context you think are required to answer this question into the input for this tool. And do not edit the response of this tool at all. It should be passed as is to the RAG tool. Do not add any extra information, explanation, or post text followup instructions. And do not display any pretext fluff or post text followups.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The user's query properly framed with all necessary context.",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "RAG_Tool",
                "description": "The finest tool to find the correct and grounded answers for your queries. Use it to find relevant papers, articles, and other documents that can help answer the user's question. DO NOT CALL THIS TOOL DIRECTLY. CALL APPROXIMATE TOOL BEFORE THIS TOOL. **ONLY PASS THE OUTPUT OF APPROXIMATE TOOL AS <query> INPUT TO THIS TOOL**. Do not change anything. Do not add or remove anything. Do not reframe the query. **INPUT = OUTPUT FROM APPROXIMATE TOOL. THATS IT.** Do not add any extra information, explanation, or post text followup instructions. **NO PRETEXT FLUFF OR POST TEXT FOLLOWUPS** to the input query. Assume that the results from this tool are 100% correct and are the ground truth. Along with the RAG searchable query, also pass the question so that another model can understand what it has to do with this information. If the user asks a factual question, you MUST use this tool to answer it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The output of the approxAnswer tool, verbatim. No changes.",
                        },
                        "question": {
                            "type": "string",
                            "description": "The original question asked by the user. Well written and structured form of the question. No fluff or extra information. Just the question. Write the question well enough and with enough context for the next model to understand what it has to do with this information returned by the RAG tool.",
                        },
                    },
                    "required": ["query", "question"],
                },
            },
        },
    ]

    functions = {
        "approxAnswer": approxAnswer,
        "RAG_Tool": RAG_Tool,
    }
    return tools, functions
