from dotenv import load_dotenv
from langchain.agents import(
    create_react_agent,
    AgentExecutor
)
from langchain_openai import ChatOpenAI
from typing import Any

load_dotenv()

from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain import hub

def main():
    print("Starting ...")
    
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
    
    
    tools = [PythonREPLTool()]
    
    python_agent = create_react_agent(
        prompt=prompt, 
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        tools=tools,
        # verbose=True
    )
    
    python_agent_executor = AgentExecutor(agent=python_agent, tools=tools, verbose=True)
    
    # print(python_agent_executor.invoke(input=
    #                              {"input": """generate and save in current working directory 15 QRcodes
    #                             that point to www.udemy.com/course/langchain, you have qrcode package installed already"""}
    #                              ))
    
    print(python_agent_executor.invoke(
                                 {"input": """generate and save in current working directory 15 QRcodes
                                that point to www.udemy.com/course/langchain, you have qrcode package installed already"""}
                                 ))
    
        
# def python_executor_wrapper(original_prompt: str) -> dict[str, Any]:
#     return python_agent_executor.invoke({'input': original_prompt})
    
    
    
if __name__ == "__main__":
    main()
    
    
    
    
# print("Starting dotenv")