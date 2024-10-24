import openai
import os, openai
from dotenv import load_dotenv

load_dotenv()

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser


prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")

model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | model | output_parser

print(chain.invoke({"topic": "Bengali"}))

#----------------------------------------------------------------

# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
# from langchain.vectorstores import DocArrayInMemorySearch
from langchain_community.vectorstores import DocArrayInMemorySearch

vectorstore = DocArrayInMemorySearch.from_texts(
    ["Soumya works in IBM at Kolkata", "Bears like to eat honey"],
    embedding=OpenAIEmbeddings(),
)

retriever = vectorstore.as_retriever()

soumya = retriever.get_relevant_documents("where does Soumya work?")
print(soumya)

bear = retriever.get_relevant_documents("what bears eat?")
print(bear)

#----------------------------------------------------------------

template = """ Answer the question based only on following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

from langchain.schema.runnable import RunnableMap

chain = RunnableMap({
        "context": lambda x: retriever.get_relevant_documents(x["question"]),
        "question": lambda x: x["question"]                                                              
    }) | prompt | model | output_parser

answer = chain.invoke({"question": "Where does Soumya work?"})
print(answer)

#----------------------------------------------------------------

inputs = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
})

ip_s = inputs.invoke({"question": "where did Soumya work?"})
print(ip_s)

#----------------------------------------------------------------

functions = [
    {
        "name": "weather_search",
        "description": "Search for weather given a airport code",
        "parameters":{
            "type": "object",
        "properties": {
          "airport_code": {
            "type": "string",
            "description": "The airport code to get the weather for"
          },
        },
        "required": ["airport_code"]
      }
    }
  ]

prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}")
    ]
    
)

model = ChatOpenAI(temperature=0).bind(functions=functions)

runnable = prompt | model

answer_ac = runnable.invoke({"input": "What is the weather in SF?"})
print(f"\n Airport Weather Updates : /n {answer_ac}")

#----------------------------------------------------------------

functions = [
    {
      "name": "weather_search",
      "description": "Search for weather given an airport code",
      "parameters": {
        "type": "object",
        "properties": {
          "airport_code": {
            "type": "string",
            "description": "The airport code to get the weather for"
          },
        },
        "required": ["airport_code"]
      }
    },
        {
      "name": "sports_search",
      "description": "Search for news of recent sport events",
      "parameters": {
        "type": "object",
        "properties": {
          "team_name": {
            "type": "string",
            "description": "The sports team to search for"
          },
        },
        "required": ["team_name"]
      }
    }
  ]

model = model.bind(functions=functions)

runnable = prompt | model

answer_both = runnable.invoke({"input": "how did the patriots do yesterday?"})
print(f"\n Airport Weather Updates : /n {answer_both}")

#----------------------------------------------------------------

from langchain.llms import OpenAI
import json

simple_model = OpenAI(
    temperature=0, 
    max_tokens=1000, 
    model="gpt-3.5-turbo-instruct"
)
simple_chain = simple_model | json.loads

challenge = "write three poems in a json blob, where each poem is a json blob of a title, author, and first line"

simple_model.invoke(challenge)

#----------------------------------------------------------------
# simple_chain.invoke(challenge)

model = ChatOpenAI(temperature=0)
chain = model | StrOutputParser() | json.loads

chain.invoke(challenge)
final_chain = simple_chain.with_fallbacks([chain])
final_chain.invoke(challenge)

#----------------------------------------------------------------

#----------------------------------------------------------------

print("\n\n Interface with SYNC & ASYNC call : \n\n")

prompt = ChatPromptTemplate.from_template(
    "Tell me a short joke about {topic}"
)
model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | model | output_parser

resp_invoke_sync = chain.invoke({"topic": "bears"})

print("\nResponse from sync call : ", resp_invoke_sync)

async def async_call():
    resp_invoke_async = await chain.ainvoke({"topic": "bears"})
    print("\nResponse from async call : ", resp_invoke_async)
    
import asyncio
asyncio.run(async_call())

#----------------------------------------------------------------


resp_chain_batch_sync = chain.batch([{"topic": "bears"}, {"topic": "frogs"}])
print("\n\nResponse from sync batch call : \n", resp_chain_batch_sync)

for t in chain.stream({"topic": "bears"}):
    print(t)