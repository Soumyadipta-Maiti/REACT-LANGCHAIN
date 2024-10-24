from dotenv import load_dotenv
from langchain.agents import(
    create_react_agent,
    AgentExecutor
)
import langchain.tools
from langchain_openai import ChatOpenAI
from typing import Any

load_dotenv()

from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain import hub
from langchain_core.tools import Tool


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
    
    # python_agent_executor.invoke(input=
    #                              {"input": """generate and save in current working directory 15 QRcodes
    #                             that point to www.udemy.com/course/langchain, you have qrcode package installed already"""}
    #                              )
    
    # python_agent_executor.invoke(
    #                              {"input": """generate and save in current working directory 15 QRcodes
    #                             that point to www.udemy.com/course/langchain, you have qrcode package installed already"""}
    #                              )
    
    csv_agent_executor = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="episode_info.csv",
        allow_dangerous_code=True,
        verbose=True,
    )
    
    
    # print(csv_agent_executor.invoke(input={"input": "How many columns are present in episode_info.csv"}))
    
    # csv_agent_executor.invoke(
    #     input={
    #         "input": "print the seasons by ascending order of number of episodes they have in episode_info.csv"
    #     }
    # )
    
    
    def python_agent_executor_wrapper(original_prompt: str) -> dict[str, Any]:
        return python_agent_executor.invoke({'input': original_prompt})


    tools_collection = [
        Tool(name='Python Agent',
            func=python_agent_executor_wrapper,
            description="""Useful when you need to transform natual language to python and execute the
            python code, returning the result of the python code execution.
            IT DOES NOT ACCEPT PYTHON CODE AS INPUT. 
            """),
        Tool(name="CSV Agent",
            func=csv_agent_executor.invoke,
            description="""Useful when you need to answer question over episode_info.csv file.
            It takes entire question as input and return answers after running pandas calculations. 
            """)    
    ]
    
    prompts_collection = base_prompt.partial(instructions="")
    grand_agent = create_react_agent(llm=ChatOpenAI(model="gpt-4", temperature=0),
                                     tools=tools_collection,
                                     prompt=prompts_collection)
    grand_agent_executor = AgentExecutor(agent=grand_agent,
                                          tools=tools_collection,
                                          verbose=True)
    
    # print(grand_agent_executor.invoke(input={
    #     "input": "which season has the most episodes within episode_info.csv file?"
    # }))
    
    # print(grand_agent_executor.invoke(input={
    #     "input": "List All distinct writers within episode_info.csv file?"
    # }))
    
    print(grand_agent_executor.invoke(input={
        "input": "Generate and save in QR-CODES sub-folder of current working directory 4 qrcodes that point to `https://www.udemy.com/course/langgraph/`"
        }))
    
    
if __name__ == "__main__":
    main()
    
    
    
    
# print("Starting dotenv")