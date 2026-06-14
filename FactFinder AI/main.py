from dotenv import load_dotenv
import os
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import save_tool, date_tool, search_tool, wiki_tool, yahoo_news_tool

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


llm = ChatOpenAI(model="gpt-4o-mini")

response = llm.invoke("What are the latest developments in AI research?")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """ You are a research assistant that gathers information on
                a given topic using various tools and provides a summary of
                the findings along with sources and tools used.
                
                Wrap the output in this formate and provide no other text\n{format_instructions}""",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [save_tool, date_tool, search_tool, wiki_tool, yahoo_news_tool]
agent = create_tool_calling_agent(llm=llm, 
                                  tools=tools, 
                                  prompt=prompt
                                  )

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("What can i help you research today?\n\nQuery:")
raw_response = agent_executor.invoke({"query": query})

try:
    structured_response = parser.parse(raw_response.get("output")[0]["text"])
    print("\nStructured Response:")
except Exception as e:
    print(f"Error parsing structured response: {e}, Raw response: {raw_response}")