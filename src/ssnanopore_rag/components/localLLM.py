from copy import deepcopy
import logging

import ollama
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text

logger = logging.getLogger(__name__)

THEME = "#d79921"
MUTED = "#7c6f64"

console = Console()


def ask_user() -> str:
    console.print("")
    console.print("_" * console.width, style=f"dim {THEME}")
    console.print("")
    return console.input(Text("You ❯ ", style=f"bold {THEME}"))


def _thinking_panel(text: str) -> Padding:
    return Padding(
        Panel(
            Text(text.strip(), style=f"dim italic {MUTED}"),
            title="thinking",
            title_align="left",
            border_style=f"dim {MUTED}",
            padding=(0, 1),
        ),
        (0, 0, 0, 6),
    )


def render_stream(stream, display: bool = True) -> tuple[str, list]:
    """Consume an ollama stream: Returns (answer, tool_calls)."""
    thinking, answer, tool_calls = "", "", []
    with Live(console=console, refresh_per_second=16, vertical_overflow="visible") as live:
        for chunk in stream:
            if chunk.message.thinking:
                thinking += chunk.message.thinking
            if chunk.message.content:
                answer += chunk.message.content
            if chunk.message.tool_calls:
                tool_calls.extend(chunk.message.tool_calls)
            parts = []
            if thinking:
                parts.append(_thinking_panel(thinking))
            if answer:
                parts.append(Markdown(answer))
            if display:
                live.update(Group(*parts))
    return answer, tool_calls


def welcome():
    text = """
  _                     _   _____            _____
 | |                   | | |  __ \     /\   / ____|
 | |     ___   ___ __ _| | | |__) |   /  \ | |  __
 | |    / _ \ / __/ _` | | |  _  /   / /\ \| | |_ |
 | |___| (_) | (_| (_| | | | | \ \  / ____ \ |__| |
 |______\___/ \___\__,_|_| |_|  \_\/_/    \_\_____|


    """
    console.print(
        Panel(
            Text(text, style=f"bold {THEME}", justify="left"),
            title="Welcome to the future!",
            title_align="left",
            border_style=f"dim {THEME}",
            padding=(0, 1),
        )
    )


class LLM:
    def __init__(
        self,
        model: str = "gemma4:e2b",
        tools: list[dict] = None,
        functions: dict[str, callable] = None,
    ) -> None:
        if functions is None:
            functions = {}
        if tools is None:
            tools = []
        self.model = model
        self.msgs = [
            {
                "role": "system",
                "content": """
You are genius scientist. You are able to understand and answer questions related to nanosciece, nanopores, biophysics, and electronics. Answer only if you are sure. You use all the tools at your disposal to answer the question whenever needed. Use reasoning, step by step logic and internal monologue to answer the question. Always assume that tools know better than you and using them increases the chances of getting the correct answer. And using the RAG tool for any query is mandatory to answer the question (unless its irrelevant to the question and you think it will decrease the chances of getting the correct answer). If the asked question is out of your context and too irrelevant, then just say I can't answer that question. Your main goal is to only give correct answers to the user's questions. If you are not sure about any information, you mention that in the answer and say I don't know or I'm not sure under given context and searched documents.""".strip(),
            }
        ]
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

    def call(self, query: str, display: bool = True) -> str:
        self.msgs.append({"role": "user", "content": query})
        for _ in range(self.MAX_TOOL_CALLS):
            stream = ollama.chat(
                model=self.model,
                messages=self.msgs,
                tools=self.tools,
                options=self.options,
                keep_alive=self.keep_alive,
                think=True,
                stream=True,
            )

            answer, tool_calls = render_stream(stream, display)

            self.msgs.append({"role": "assistant", "content": answer, "tool_calls": tool_calls})
            if not tool_calls:
                return answer
            self.execute_tool_calls(tool_calls)

        return self.msgs[-1]["content"]

    def execute_tool_calls(self, tool_calls: list[dict]) -> str:
        for tool in tool_calls:
            console.print(f"[bold {THEME}]Running {tool.function.name} tool...[/]", style="dim")
            if tool.function.name in self.functions:
                result = str(self.functions[tool.function.name](**tool.function.arguments))
                self.msgs.append({"role": "tool", "content": result})
                # logger.debug(self.msgs[-1]["content"])
            else:
                self.msgs.append({"role": "tool", "content": "Tool not found"})
                logger.warning(self.msgs[-1]["content"])


def single_turn_llm(query: str, llm_instance: LLM, with_history: bool = False) -> str:
    messages = deepcopy(llm_instance.msgs) if with_history else []
    messages.append({"role": "user", "content": query})
    llm = LLM()
    llm.msgs = messages
    output = llm.call(query, display=False)
    return output


def debug_get_tools_and_functions() -> tuple[list, dict]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                },
            },
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
                        "b": {"type": "number"},
                    },
                },
            },
        },
    ]
    functions = {"add": lambda a, b: a + b, "subtract": lambda a, b: a - b}
    return tools, functions


def main():
    welcome()
    tools, functions = debug_get_tools_and_functions()
    llm = LLM(
        tools=tools,
        functions=functions,
    )
    while (query := ask_user().strip()) not in ("", "exit", "quit"):
        llm.call(query)

    print("\n\n" + "=" * console.width)
    print(single_turn_llm("summarize our conversation", llm, with_history=False))


if __name__ == "__main__":
    main()
