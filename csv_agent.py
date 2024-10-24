from dotenv import load_dotenv
from langchain.agents import(
    create_react_agent,
    AgentExecutor
)
from langchain_openai import ChatOpenAI
from typing import Any

load_dotenv()

from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain import hub

def main():
    print("Starting for CSV Agent ...")
    
    instr_python_agent = """
    You are a agent to write & execute code in python to answer questions.
    You have access to Python REPL that you use to execute python code.
    You have qrcode package installed.
    If you have any error, debug & try again to fix the error.
    Only use output of your code to answer the question.
    You might know the answer without running any code, but you should run the code to get answer.
    If it seems like you can't write code to answer the question, just return "I don't know" as answer.
    """
    
    
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    
    prompt = base_prompt.partial(instructions=instr_python_agent)
    
    # tools = [csv_agent]
    
    csv_agent_executor = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="episode_info.csv",
        allow_dangerous_code=True,
        verbose=True,
    )
    
    # csv_agnet_executor = AgentExecutor(agent=csv_agent, verbose=True, tools=)
    
    # print(csv_agent_executor.invoke(input={"input": "How many columns are present in episode_info.csv"}))
    
    print(csv_agent_executor.invoke(
        input={
            "input": "print the seasons by ascending order of number of episodes they have in episode_info.csv"
        }
    ))
    
    
    
    
if __name__ == "__main__":
    main()