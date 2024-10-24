from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import datetime
import requests
import wikipedia
from typing import List
from langchain_openai import ChatOpenAI
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool
from langchain.tools.render import format_tool_to_openai_function
# from langchain.tools.render import convert_to_openai_function
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.schema.runnable import RunnablePassthrough

# Initialize FastAPI app
app = FastAPI()

# Define input schema for the temperature API
class OpenMeteoInput(BaseModel):
    latitude: float
    longitude: float

# FastAPI route for fetching the temperature
@app.post("/get_current_temperature")
def get_current_temperature(data: OpenMeteoInput):
    """Fetch current temperature for given coordinates."""
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    params = {
        'latitude': data.latitude,
        'longitude': data.longitude,
        'hourly': 'temperature_2m',
        'forecast_days': 1,
    }

    response = requests.get(BASE_URL, params=params)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="API request failed")

    results = response.json()
    current_utc_time = datetime.datetime.utcnow()
    time_list = [datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00')) for time_str in results['hourly']['time']]
    temperature_list = results['hourly']['temperature_2m']
    
    closest_time_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - current_utc_time))
    current_temperature = temperature_list[closest_time_index]
    
    return {"temperature": current_temperature}

# FastAPI route for Wikipedia search
@app.get("/search_wikipedia")
def search_wikipedia(query: str):
    """Search Wikipedia and return page summaries."""
    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles[:3]:
        try:
            wiki_page = wikipedia.page(title=page_title, auto_suggest=False)
            summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
        except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError):
            continue
    if not summaries:
        return {"result": "No good Wikipedia search result found"}
    return {"summaries": summaries}

# FastAPI route for the custom function
@app.get("/create_your_own")
def create_your_own(query: str):
    """Reverse the string provided."""
    return {"reversed_query": query[::-1]}

# Create chatbot functionality with FastAPI POST route
class ChatInput(BaseModel):
    query: str

# tools = [get_current_temperature, search_wikipedia, create_your_own]

# Convert functions to OpenAI-compatible tools
tools = [
    convert_to_openai_function(get_current_temperature),
    convert_to_openai_function(search_wikipedia),
    convert_to_openai_function(create_your_own)
]

@app.post("/chatbot")
def chatbot(query: ChatInput):
    """Handle chatbot queries."""
    
    # model = ChatOpenAI(temperature=0).bind(functions=[format_tool_to_openai_function(f) for f in tools])
    
        # Use the new function `convert_to_openai_function()`
    functions = [convert_to_openai_function(f) for f in tools]
    model = ChatOpenAI(temperature=0).bind(functions=functions)
    
    # memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
    # Update: Memory usage as per new version
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="input", return_messages=True)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful but sassy assistant"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    chain = RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_to_openai_functions(x["intermediate_steps"])
    ) | prompt | model | OpenAIFunctionsAgentOutputParser()
    
    qa = AgentExecutor(agent=chain, tools=tools, verbose=False, memory=memory)
    result = qa.invoke({"input": query.query})
    
    return {"response": result['output']}

# FastAPI root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI Chatbot API!"}

