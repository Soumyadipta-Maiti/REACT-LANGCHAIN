import os
import requests
import datetime
import wikipedia
import panel as pn
import param
import whylogs as why
from langkit import llm_metrics
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents import AgentExecutor
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents.format_scratchpad import format_to_openai_functions
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()



#------------------------------------------------------------------


# Define the input schema for the weather API
class OpenMeteoInput(BaseModel):
    latitude: float = Field(..., description="Latitude of the location to fetch weather data for")
    longitude: float = Field(..., description="Longitude of the location to fetch weather data for")

@tool(args_schema=OpenMeteoInput)
def get_current_temperature(latitude: float, longitude: float) -> dict:
    """Fetch current temperature for given coordinates."""
    
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    
    # # Log the input data
    # with logger.log():
    #     logger.log_dataframe(pd.DataFrame([{"latitude": latitude, "longitude": longitude}]))
    
    # Parameters for the request
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': 'temperature_2m',
        'forecast_days': 1,
    }

    # Make the request
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        results = response.json()
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")

    current_utc_time = datetime.datetime.utcnow()
    time_list = [datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00')) for time_str in results['hourly']['time']]
    temperature_list = results['hourly']['temperature_2m']
    
    closest_time_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - current_utc_time))
    current_temperature = temperature_list[closest_time_index]
    
    # # Log the output data
    # with logger.log():
    #     logger.log_dataframe(pd.DataFrame([{"current_temperature": current_temperature}]))
    
    return f'The current temperature is {current_temperature}°C'

@tool
def search_wikipedia(query: str) -> str:
    """Run Wikipedia search and get page summaries."""
    
    # # Log the input query
    # with logger.log():
    #     logger.log_dataframe(pd.DataFrame([{"query": query}]))
    
    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles[: 3]:
        try:
            wiki_page = wikipedia.page(title=page_title, auto_suggest=False)
            summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
        except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError):
            pass
    
    if not summaries:
        output = "No good Wikipedia Search Result was found"
    else:
        output = "\n\n".join(summaries)
    
    # # Log the output
    # with logger.log():
    #     logger.log_dataframe(pd.DataFrame([{"output": output}]))
    
    return output

@tool
def create_your_own(query: str) -> str:
    """Example custom tool function."""
    
    # # Log the input query
    # with logger.log():
    #     logger.log_dataframe(pd.DataFrame([{"query": query}]))
    
    result = query[::-1]  # Reverse the string for illustration

    # # Log the output
    # with logger.log():
    #     logger.log_dataframe(pd.DataFrame([{"output": result}]))
    
    return result

class cbfs(param.Parameterized):
    
    def __init__(self, tools, **params):
        super(cbfs, self).__init__(**params)
        self.panels = []
        self.functions = [format_tool_to_openai_function(f) for f in tools]
        self.model = ChatOpenAI(temperature=0).bind(functions=self.functions)
        self.memory = ConversationBufferMemory(return_messages=True,memory_key="chat_history")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful but sassy assistant"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        self.chain = RunnablePassthrough.assign(
            agent_scratchpad = lambda x: format_to_openai_functions(x["intermediate_steps"])
        ) | self.prompt | self.model | OpenAIFunctionsAgentOutputParser()
        self.qa = AgentExecutor(agent=self.chain, tools=tools, verbose=False, memory=self.memory)
    
    def convchain(self, query):
        if not query:
            return
        inp.value = ''
        result = self.qa.invoke({"input": query})
        
        
        self.answer = result['output']
        
        # # Log the query and answer
        # with logger.log():
        #     logger.log_dataframe(pd.DataFrame([{"query": query, "answer": self.answer}]))
        
        self.panels.extend([
            pn.Row('User:', pn.pane.Markdown(query, width=450)),
            pn.Row('ChatBot:', pn.pane.Markdown(self.answer, width=450, styles={'background-color': '#F6F6F6'}))
        ])
        return pn.WidgetBox(*self.panels, scroll=True)


    def clr_history(self, count=0):
        self.chat_history = []
        return 

tools = [get_current_temperature, search_wikipedia, create_your_own]

# GUI Setup
pn.extension()
cb = cbfs(tools)
inp = pn.widgets.TextInput(placeholder='Enter your query here …')

conversation = pn.bind(cb.convchain, inp) 

tab1 = pn.Column(
    pn.Row(inp),
    pn.layout.Divider(),
    pn.panel(conversation, loading_indicator=True, height=400),
    pn.layout.Divider(),
)

dashboard = pn.Column(
    pn.Row(pn.pane.Markdown('Q&A Chat-Bot')),
    pn.Tabs(('Conversation', tab1))
)

# Start the Panel server to render the chatbot dashboard
dashboard.show(port=5020)

#----------------------------------------------------------------

