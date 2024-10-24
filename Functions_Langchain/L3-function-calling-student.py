import os, openai
from dotenv import load_dotenv
load_dotenv()

from typing import List
from pydantic import BaseModel, Field

class User():
    def __init__(self, name: str, age: int, email: str):
        self.name = name
        self.age = age
        self.email = email
        
foo = User(name='Joe', age=32, email='joe@gmail.com')

print(foo.name)

hoo = User(name='Hoe', age='mid', email='joe@gmail.com')
print(hoo.age)

class pUser(BaseModel):
    name: str
    age: int
    email: str
    
poo = pUser(name='Poo', age=32, email='joe@gmail.com')
print(poo.name)

# moo=pUser(name='Moo', age='old', email='joe@another.com')
# print(moo.age)

class Class(BaseModel):
    students: list[pUser]
    
Moo = Class(students=[pUser(name='Moo', age=55, email='joe@another.com')])

print(Moo)


#----------------------------------------------------------------

# Pydantic to OpenAI function definition

class WeatherSearch(BaseModel):
    """Call this with an airport code to get the weather at that airport"""
    airport_code: str = Field(description="airport code to get the weather for")

from langchain.utils.openai_functions import convert_pydantic_to_openai_function

weather_function = convert_pydantic_to_openai_function(WeatherSearch)
print (weather_function)
                   
    
class WeatherSearch1(BaseModel):
    airport_code: str = Field(description="airport code to get weather for")
    
weather_function_1 = convert_pydantic_to_openai_function(WeatherSearch1)
print (weather_function_1)

class WeatherSearch2(BaseModel):
    """Call this with an airport code to get the weather at that airport"""
    airport_code: str
    
weather_function_2 = convert_pydantic_to_openai_function(WeatherSearch2)
print (weather_function_2)

#----------------------------------------------------------------

from langchain_openai import ChatOpenAI
# from Functions_Langchain.L1-openai_functions_student import functions

model = ChatOpenAI()
resp_wtf = model.invoke("What is the weather in SF today?", functions=[weather_function])
print(resp_wtf)

model_with_function = model.bind(functions=[weather_function])

resp_wtf_bind = model_with_function.invoke("what is the weather in sf?")
print(resp_wtf_bind)

#----------------------------------------------------------------

# Forcing it to use a function

model_with_forced_function = model.bind(functions=[weather_function], function_call={"name":"WeatherSearch"})

resp_wtf_bind_fc = model_with_forced_function.invoke("what is the weather in sf?")
print(f"\n\n Forcing it to use a function \n {resp_wtf_bind_fc}")

#----------------------------------------------------------------

# Using in a chain

from langchain.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ('system', "You are a helpful assistant"),
    ("user", "{input}")    
])

chain = prompt | model_with_function

resp_model_wf_chain = chain.invoke({"input": "what is the weather in sf?"})
print(f"\n\n Using in a chain \n {resp_model_wf_chain}")

#----------------------------------------------------------------

# Using multiple functions

class ArtistSearch(BaseModel):
    """Call this to get the names of songs by a particular artist"""
    artist_name: str = Field(description="name of artist to look up")
    n: int = Field(description="number of results")
    
functions = [
    convert_pydantic_to_openai_function(ArtistSearch),
    convert_pydantic_to_openai_function(WeatherSearch)
]

model_with_mult_functions = model.bind(functions=functions)

resp_model_wmf_chain = model_with_mult_functions.invoke("what is the weather in sf?")
print(f"\n\n Using multiple functions \n {resp_model_wmf_chain}")

resp_model_wmf_chain = model_with_mult_functions.invoke("what are three songs by taylor swift?")
print(f"\n\n Using multiple functions \n {resp_model_wmf_chain}")

resp_model_wmf_chain = model_with_mult_functions.invoke("hi!")
print(f"\n\n Using multiple functions \n {resp_model_wmf_chain}")

#----------------------------------------------------------------




    
