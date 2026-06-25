import logging
import ollama

logging.basicConfig(
    level=logging.INFO, 
    format='%(levelname)s:%(name)s:%(lineno)d: %(message)s'
)
logger = logging.getLogger(__name__)

class LLM:
    def __init__(
        self,
        model: str = "gemma4:e2b",
        tools: list[dict] = [],
        functions: dict[str, callable] = {}
    ) -> None:
        self.model = model
        self.msgs = [{
            "role": "system", 
            "content": "You are genius scientist. You are able to understand and answer questions related to nanosciece, nanopores, biophysics, and electronics. Answer only if you are sure. You use all the tools at your disposal to answer the question whenever needed. Use reasoning, step by step logic and internal monologue to answer the question. Always assume that tools know better than you and using them increases the chances of getting the correct answer. You life's goal is to only give correct answers to the user's questions. If you are not sure about any information, you mention that in the answer and say I don't know or I'm not sure under given context and searched documents."}]
        self.options = {
            "temperature": 0.5,
            "top_k": 64,
            "top_p": 0.95,
            "repeat_penalty": 1.1,
        }
        self.keep_alive = "10m"
        self.tools = tools
        self.functions = functions
        self.MAX_TOOL_CALLS = 50

    def call(self, query: str) -> str:
        self.msgs.append({"role": "user", "content": query})
        for _ in range(self.MAX_TOOL_CALLS):
            res = ollama.chat(model=self.model, messages=self.msgs, tools=self.tools, options=self.options, keep_alive=self.keep_alive)
            self.msgs.append(res["message"])
            logger.info(self.msgs[-1]["content"])
            if not res.message.tool_calls:
                return res.message.content
            for tool in res.message.tool_calls:
                if tool.function.name in self.functions:
                    self.msgs.append({"role": "tool", "content": str(self.functions[tool.function.name](**tool.function.arguments))})
                    logger.info(self.msgs[-1]["content"])
                else:
                    self.msgs.append({"role": "tool", "content": "Tool not found"})
                    logger.warn(self.msgs[-1]["content"])
            
        return self.msgs[-1]["content"]
        
def main():
    llm = LLM(tools=[
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"}
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "subtract",
                "description": "Subtract two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"}
                    }
                }
            }
        }
    ],
    functions={
        "add":lambda a, b: a + b,
        "subtract":lambda a, b: a - b
    })
    llm.call("Call all the tools you have and debug that they are working properly. Always give response after tool calls. And show your work step by step. Use reasoning and internal monologue to answer the question.")
    

if __name__ == "__main__":
    main()
