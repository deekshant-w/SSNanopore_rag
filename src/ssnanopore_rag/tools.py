from .components.localLLM import single_turn_llm
from .prepare import prepareDatabase


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
        self.count = 2
        self.qCount = self.count
        self.cCount = self.count
        self.pCount = self.count

    def call(self, query: str):
        qPicks = self.qdrantStore_Rerank.query([query], n_results=self.qCount)
        qDocIds = [i.payload["doc_id"] for i in qPicks.points]
        # print(f"QdrantStore_Rerank: {qPicks}")
        print(f"{qDocIds=}")

        cPicks = self.chromaStore.query([query], n_results=self.cCount)
        cDocIds = [i["doc_id"] for i in cPicks["metadatas"][0]]
        # print(f"ChromaStore: {cPicks}")
        print(f"{cDocIds=}")

        pPicks = self.pineconeStore_Dense.query([query], n_results=self.pCount)
        pDocIds = [i.metadata["doc_id"] for i in pPicks.matches]
        # print(f"PineconeStore_Dense: {pPicks}")
        print(f"{pDocIds=}")


__RAG_Tool = _RAG_Tool()
RAG_Tool = __RAG_Tool.call


def get_tools_and_functions() -> tuple[list[dict], dict]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "approxAnswer",
                "description": "Get an approximate answer to a question before asking the RAG tool. Use this tool strictly before calling the RAG tool. The output of this tool should be the input to the RAG tool.",
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
