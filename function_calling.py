from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from langchain.agents import create_tool_calling_agent, AgentExecutor


@tool
def multiply(a: float, b: float)->float:
    """Multiply 'a' times 'b'."""
    return a * b






if __name__ == '__main__':
    print("Starting Function Calling ...")
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a hekpful assistant"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    
    tools = [TavilySearchResults(), multiply]
    llm = ChatOpenAI(model="gpt-4")
    # llm=ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    
    res = agent_executor.invoke(input={
        'input': "what is the weather in dubai right now? compare it with San Fransisco, output should in in celsious?"})
    
    print(res)