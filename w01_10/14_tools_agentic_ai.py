# What Makes a Tool in Agentic AI – Demo Build
# Goal: Learn how to design, implement, and expose a tool that an agent can call.

from langchain_core.tools import Tool
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)



# 1. Calculator
def calculator(expr: str):
    try:
        return str(eval(expr))
    except Exception as e:
        return f"Error: {e}"

# 2. Joke generator
def joke(topic: str):
    return f"Here’s a {topic}-themed joke: Why did the {topic} cross the road? To learn Agentic AI!"

# 3. Location (fake API call using Open-Meteo as demo)
def location(city: str):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
    r = requests.get(url, timeout=10).json()
    if "results" not in r:
        return f"No location info for {city}"
    coords = r["results"][0]
    return f"{city} located at lat {coords['latitude']}, lon {coords['longitude']} (demo response)."


tools = [
    Tool(name="Calculator", func=calculator, description="Evaluate math expressions."),
    Tool(name="Joke", func=joke, description="Tell a quick joke about a topic."),
    Tool(name="Location", func=location, description="Get demo location info for a city."),
]


agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant with access to tools. Use them whenever they are useful.",
)

for prompt in [
    # "What is 12 * (7+3)?",
    # "Tell me a joke about robots.",
    "Where is Dombivali located?",
    "How is weather in New York City?",
]:
    response = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    print(f"User: {prompt}")
    print(response["messages"][-1].content)
    print("-" * 40)



