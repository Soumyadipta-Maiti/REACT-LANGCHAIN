import openai
import os, openai
from dotenv import load_dotenv

load_dotenv()

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser


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
    
# response = await chain.ainvoke({"topic": "bears"})
# response

