import os
import openai

from dotenv import load_dotenv, find_dotenv

from typing import List
from pydantic import BaseModel, Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_function


class Tagging(BaseModel):
    """Tag the piece of text with particular info."""
    sentiment: str = Field(description="sentiment of text, should be `pos`, `neg`, or `neutral`")
    language: str = Field(description="language of text (should be ISO 639-1 code)")
    
tagging_openai = convert_pydantic_to_openai_function(Tagging)
print(f"\n Tagging function compatible with OpenAI: \n{tagging_openai}")

#----------------------------------------------------------------

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
# from Functions_Langchain.L1-openai_functions_student import functions

model = ChatOpenAI(temperature=0)

tagging_functions = [convert_pydantic_to_openai_function(Tagging)]

prompt = ChatPromptTemplate.from_messages([
    ("system", "Think carefully, and then tag the text as instructed"),
    ("user", "{input}")
])

model_with_functions = model.bind(
    functions=tagging_functions,
    function_call={"name": "Tagging"}
)

tagging_chain = prompt | model_with_functions

resp_fn_chain_1 = tagging_chain.invoke({"input": "I love Langchain."})
print(f"\n\nTagging with Function Calling for English: \n{resp_fn_chain_1}")

# resp_fn_chain_2 = tagging_chain.invoke({"input": "non mi piace questo cibo"})
resp_fn_chain_2 = tagging_chain.invoke({"input": "amar aaj kichu bhalo lagche na"})
print(f"\n\nTagging with Function Calling for Bengali : \n{resp_fn_chain_2}")

#----------------------------------------------------------------

from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
tagging_chain_json = prompt | model_with_functions | JsonOutputFunctionsParser()

resp_fn_chain_3 = tagging_chain.invoke({"input": "kaha ho tum"})
print(f"\n\nTagging with Function Calling for Hindi: \n{resp_fn_chain_3}")

#----------------------------------------------------------------

# Extraction

from typing import Optional
class Person(BaseModel):
    """Information about a person."""
    name: str = Field(description="person's name")
    age: Optional[int] = Field(description="person's age")
    
    
class Information(BaseModel):
    """Information to extract."""
    people: List[Person] = Field(description="List of info about people")
    
convert_pydantic_to_openai_function(Information)

extraction_functions = [convert_pydantic_to_openai_function(Information)]
extraction_model = model.bind(functions=extraction_functions, function_call={"name": "Information"})

resp_fn_chain_4 = extraction_model.invoke("Joe is 30, his mom is Martha")
print(f"\n\nExtraction with Function Calling - 1: \n{resp_fn_chain_4}")


#----------------------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract the relevant information, if not explicitly provided do not guess. Extract partial info"),
    ("human", "{input}")
])

extraction_chain = prompt | extraction_model
resp_fn_chain_5 = extraction_chain.invoke({"input": "Joe is 30, his mom is Martha"})
print(f"\n\nExtraction with Function Calling + chain - 1: \n{resp_fn_chain_5}")


extraction_chain = prompt | extraction_model | JsonOutputFunctionsParser()
resp_fn_chain_6 = extraction_chain.invoke({"input": "Joe is 30, his mom is Martha"})
print(f"\n\nExtraction with Function Calling + chain +json - 1: \n{resp_fn_chain_6}")

from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
extraction_chain = prompt | extraction_model | JsonKeyOutputFunctionsParser(key_name="people")
resp_fn_chain_7 = extraction_chain.invoke({"input": "Joe is 30, his mom is Martha"})
print(f"\n\nExtraction with Function Calling + chain + json + key - 1: \n{resp_fn_chain_7}")

#----------------------------------------------------------------

# Doing it for real

from langchain.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
documents = loader.load()

doc = documents[0]
page_content = doc.page_content[:10000]

print(f"\n\n Pages from web content \n{page_content[:1000]}")

class Overview(BaseModel):
    """Overview of a section of a text."""
    summary: str = Field(description="Provide a concise summary of the content.")
    language: str = Field(description="Provide the language that the content is written in.")
    keywords: str = Field(description="Provide keywords related to the content.")
    
overview_tagging_function = [
    convert_pydantic_to_openai_function(Overview)
]

tagging_model = model.bind(
    functions=overview_tagging_function,
    function_call={"name":"Overview"}
)

tagging_chain = prompt | tagging_model | JsonOutputFunctionsParser()

resp_fn_chain_page_1 = tagging_chain.invoke({"input": page_content})
print(f"\n\n Extraction of Web-Page Summary with Function Calling + chain + json - 1: \n{resp_fn_chain_page_1}")

#----------------------------------------------------------------

class Paper(BaseModel):
    """Information about papers mentioned."""
    title: str
    author: Optional[str]


class Info(BaseModel):
    """Information to extract"""
    papers: List[Paper]
    
paper_extraction_function = [
    convert_pydantic_to_openai_function(Info)
]

print(f"\n\npaper_extraction_function content : \n{paper_extraction_function}")

extraction_model = model.bind(
    functions=paper_extraction_function, 
    function_call={"name":"Info"}
)
extraction_chain = prompt | extraction_model | JsonKeyOutputFunctionsParser(key_name="papers")

resp_fn_chain_page_2 = extraction_chain.invoke({"input": page_content})
print(f"\n\n Extraction of Web-Page Summary with Function Calling + chain + Key + json - 1: \n{resp_fn_chain_page_2}")

#----------------------------------------------------------------

template = """A article will be passed to you. Extract from it all papers that are mentioned by this article. 
Do not extract the name of the article itself. If no papers are mentioned that's fine - you don't need to extract any! 
Just return an empty list. Do not make up or guess ANY extra information. Only extract what exactly is in the text."""

prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", "{input}")
])

extraction_chain = prompt | extraction_model | JsonKeyOutputFunctionsParser(key_name="papers")
resp_fn_chain_page_3 = extraction_chain.invoke({"input": page_content})
print(f"\n\n Extraction of Web-Page Summary with Function Calling + chain + Key + json - 2: \n{resp_fn_chain_page_2}")

resp_fn_chain_page_3 = extraction_chain.invoke({"input": "hi"})
print(f"\n\n Extraction of Hi with Function Calling + chain + Key + json - 2: \n{resp_fn_chain_page_3}")

#----------------------------------------------------------------
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_overlap=0)

splits = text_splitter.split_text(doc.page_content)
length = len(splits)
print(f"\n\n Length of RecursiveCharacterTextSplitter : {length}")

def flatten(matrix):
    flat_list = []
    for row in matrix:
        flat_list += row
    return flat_list

flatten([[1,2], [3,4]])

print(f"\n\nContent of first Split : \n{splits[0]}")

#----------------------------------------------------------------
from langchain.schema.runnable import RunnableLambda
prep = RunnableLambda(
    lambda x: [{"input": docs} for docs in text_splitter.split_text(x)]
)

# prep.invoke("hi")

chain = prep | extraction_chain.map() | flatten
resp_fn_chain_page_runnable_1 = chain.invoke(doc.page_content)
print(f"\n\n Extraction of Hi with Function Calling + chain + flatten - 2: \n{resp_fn_chain_page_runnable_1}")
