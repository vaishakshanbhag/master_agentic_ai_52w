"""
Lab 2: Building Your First Custom Agent Tool
Goal: Design and implement a reusable custom tool (name, description, input schema, function),
add validation + error handling + caching, and expose it to an agent.
"""

import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from tools.url_summarizer import summarize_url

load_dotenv()

# --- Tool 1: URL Summarizer ---
class UrlSummarizerInput(BaseModel):
    """Schema for UrlSummarizer tool inputs.

    Attributes:
        url: The URL to summarize.
        max_words: Maximum number of words to include in the summary.
    """

    url: str = Field(description="The URL to summarize")
    max_words: int = Field(default=120, description="Max words in summary")

def _summarize(url: str, max_words: int = 120):
    """Summarize the content of a web page using the helper function.

    Args:
        url: The web page URL to fetch and summarize.
        max_words: Maximum number of words in the returned summary.

    Returns:
        A concise summary of the page content.
    """
    return summarize_url(url=url, max_words=max_words)

url_tool = StructuredTool.from_function(
    func=_summarize,
    name="UrlSummarizer",
    description=
        "REQUIRED for summarizing any web page or URL. Fetches the live page content "
        "and returns an accurate summary. Always use this instead of guessing from memory, "
        "since page content may have changed or you may not have seen it.",
    args_schema=UrlSummarizerInput,
)

# --- Tool 2: Calculator ---

def calculator(expr: str) -> str:
    """Evaluate a math expression and return the result as a string.

    This tool is deliberately conservative in output formatting: it returns the
    numeric result as a string or an error message if evaluation fails.

    Args:
        expr: A mathematical expression to evaluate.

    Returns:
        The evaluated result as a string, or an error description.
    """
    try:
        return str(eval(expr))
    except Exception as e:
        return f"Error: {e}"

calc_tool = StructuredTool.from_function(
    func=calculator,
    name="Calculator",
    description="Evaluate math expressions like '12*(7+3)'.",
)

# --- LLM + Agent ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [url_tool, calc_tool]

agent = create_agent(
    llm,
    tools,
    system_prompt=(
        "You must use the UrlSummarizer tool whenever asked to summarize a URL, "
        "even if you think you already know the content. Never answer from your own knowledge "
        "for summarization or calculation tasks — always call the appropriate tool."
    ),
)

if __name__ == "__main__":
    # result = agent.invoke({
    #     "messages": [
    #         {"role": "user", "content": "Use UrlSummarizer with {'url':'https://openai.com/research', 'max_words': 120}"}
    #     ]
        
    # })
    # # result["messages"] contains the full conversation; last message is the agent's final answer
    # print(result["messages"][-1].content)
    result1 = agent.invoke({
    "messages": [
        {"role": "user", "content": (
            "Summarize this page in ~80 words using UrlSummarizer: "
            "{'url':'https://lilianweng.github.io/posts/2023-06-23-agent/','max_words':80}"
        )}
    ]
    })
    # print(result1["messages"][-1].content)

    # Print the full conversation messages to inspect how the tool call was generated.
    for msg in result1["messages"]:
        print(f"--- {type(msg).__name__} ---")
        print(msg)
        print()

    # Uncomment the block below to test the Calculator tool as a second request.
    # result2 = agent.invoke({
    #     "messages": [
    #         {"role": "user", "content": "Also, what is 25*(8-3)? Use Calculator."}
    #     ]
    # })
    # # print(result2["messages"][-1].content)
    # for msg in result2["messages"]:
    #     print(f"--- {type(msg).__name__} ---")
    #     print(msg)
    #     print()