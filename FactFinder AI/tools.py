from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun, YahooFinanceNewsTool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime

def save_to_file(data: str, filename: str = "research_findings.txt"):
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"----- Research Findings ----- \nTImestamps: {timestamp}\n\n{data}"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Data successfully written to {filename}"


save_tool = Tool(name="Save Research Findings",
                 func=save_to_file,
                 description="Use this tool to save research findings to a file. Provide the data as a string and optionally specify a filename."
                 )

def today_date(data: str):
    return datetime.now()

date_tool = Tool(name="Get Today's Date",
                 func= today_date,
                 description="Use this tool to get today's date. No input is required."
                 )

search = DuckDuckGoSearchRun()
search_tool = Tool(name="DuckDuckGo Search",
                   func=search.run,
                   description="Use this tool to perform a search using DuckDuckGo. Provide the query as input."
                   )

wiki_app_wrapper = WikipediaAPIWrapper(top_k_results=1,
                                       doc_content_chars_max=200)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_app_wrapper)

yahoo_finance_news = YahooFinanceNewsTool()
yahoo_news_tool = Tool(name="Yahoo Finance News",
                       func=yahoo_finance_news.run,
                       description="Use this tool to get the latest news on a specific stock. Provide the stock ticker as input.")

