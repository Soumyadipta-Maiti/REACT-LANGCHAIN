import os
from langchain_openai import ChatOpenAI
import openai

from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool

from langchain.prompts import MessagesPlaceholder
from langchain.prompts import ChatPromptTemplate
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory

import requests
from pydantic import BaseModel, Field
import datetime

import panel as pn  # GUI
# import panel as pn
import param
import mlflow
import time

# Step 1: Initialize MLflow
mlflow.set_experiment("LangChain-MLflow-Chatbot")

# Define the input schema
class OpenMeteoInput(BaseModel):
    latitude: float = Field(..., description="Latitude of the location to fetch weather data for")
    longitude: float = Field(..., description="Longitude of the location to fetch weather data for")
    


@tool(args_schema=OpenMeteoInput)
def get_current_temperature(latitude: float, longitude: float) -> dict:
    """Fetch current temperature for given coordinates."""
    
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    
    # Parameters for the request
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': 'temperature_2m',
        'forecast_days': 1,
    }
    
    with mlflow.start_run(run_name="Temperature_API_Call", nested=True):
        try:
            # Make the request
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()  # Raises exception for bad requests (4XX, 5XX)
            if response.status_code == 200:
                results = response.json()
                mlflow.log_param("latitude", latitude)
                mlflow.log_param("longitude", longitude)
                mlflow.log_metric("response_status", response.status_code)
        except Exception as e:
            mlflow.log_metric("response_status", response.status_code)
            raise Exception(f"API Request failed with status code: {response.status_code} & exception: {e}")

        current_utc_time = datetime.datetime.utcnow()
        time_list = [datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00')) for time_str in results['hourly']['time']]
        temperature_list = results['hourly']['temperature_2m']
        
        closest_time_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - current_utc_time))
        current_temperature = temperature_list[closest_time_index]
        
        mlflow.log_metric("current_temperature", current_temperature)
        
        # # End the current run explicitly
        # mlflow.end_run()
        
        return f'The current temperature is {current_temperature}°C'

#----------------------------------------------------------------

import wikipedia

@tool
def search_wikipedia(query: str) -> str:
    """Run Wikipedia search and get page summaries."""
    
    with mlflow.start_run(run_name="Wikipedia_Search", nested=True):
        try:
            page_titles = wikipedia.search(query)
            summaries = []
            for page_title in page_titles[: 3]:
                try:
                    wiki_page =  wikipedia.page(title=page_title, auto_suggest=False)
                    summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
                    
                    mlflow.log_param("query", query)
                    mlflow.log_metric("pages_found", len(page_titles))
                except (
                    self.wiki_client.exceptions.PageError,
                    self.wiki_client.exceptions.DisambiguationError,
                ):
                    pass
            if not summaries:
                return "No good Wikipedia Search Result was found"
            return "\n\n".join(summaries)
        except Exception as e:
            # mlflow.end_run()  # Ensure the run ends even if there's an error
            mlflow.log_text(f"Error: {e}", "error_log.txt")
            return f"An error occurred: {e}"

#----------------------------------------------------------------


#----------------------------------------------------------------

# Create a chatbot

@tool
def create_your_own(query: str) -> str:
    """This function can do whatever you would like once you fill it in """
    print(type(query))
    return query[::-1]




#----------------------------------------------------------------

class cbfs(param.Parameterized):
    
    def __init__(self, tools, **params):
        super(cbfs, self).__init__( **params)
        self.panels = []
        self.functions = [format_tool_to_openai_function(f) for f in tools]
        self.model = ChatOpenAI(temperature=0).bind(functions=self.functions)
        self.memory = ConversationBufferMemory(return_messages=True,memory_key="chat_history")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are helpful but sassy assistant"),
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
        
        # Log the input query to MLflow
        with mlflow.start_run(run_name="Chatbot_Conversation", nested=True):
            mlflow.log_param("user_query", query)

            # Measure and log performance
            start_time = time.time()
            result = self.qa.invoke({"input": query})
            response_time = time.time() - start_time
            mlflow.log_metric("response_time", response_time)
            
            # Log model output
            self.answer = result['output'] 
            mlflow.log_text(self.answer, "output.txt")
            
            self.panels.extend([
                pn.Row('User:', pn.pane.Markdown(query, width=450)),
                pn.Row('ChatBot:', pn.pane.Markdown(self.answer, width=450, styles={'background-color': '#F6F6F6'}))
            ])
            return pn.WidgetBox(*self.panels, scroll=True)


    def clr_history(self,count=0):
        self.chat_history = []
        return 
    
#----------------------------------------------------------------

tools = [get_current_temperature, search_wikipedia, create_your_own]

pn.extension()


cb = cbfs(tools)

inp = pn.widgets.TextInput( placeholder='Enter your query here …')

conversation = pn.bind(cb.convchain, inp) 

tab1 = pn.Column(
    pn.Row(inp),
    pn.layout.Divider(),
    pn.panel(conversation,  loading_indicator=True, height=400),
    pn.layout.Divider(),
)

dashboard = pn.Column(
    pn.Row(pn.pane.Markdown('Q&A Chat-Bot')),
    pn.Tabs(('Conversation', tab1))
)

# print(dashboard)

# Start the Panel server to render the chatbot dashboard
dashboard.show(port=5015)  # You can specify any port number you prefer

#----------------------------------------------------------------

