# Tools and Routing

import os
import openai

from dotenv import load_dotenv
load_dotenv()

from langchain.agents import tool

@tool
def search(query: str) -> str:
    """Search for weather online"""
    return "42f"

print(f"search.name : {search.name}")
print(f"search.description : {search.description}")
print(f"search.args : {search.args}")

#----------------------------------------------------------------

from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="Thing to search for")
    
@tool(args_schema=SearchInput)
def search(query: str) -> str:
    """Search for weather online"""
    return "42f"

print(f"search.name : {search.name}")
print(f"search.description : {search.description}")
print(f"search.args : {search.args}")
print(f"search.run : {search.run("sf")}")

#----------------------------------------------------------------

import requests
from pydantic import BaseModel, Field
import datetime

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
    
    return f'The current temperature is {current_temperature}°C'


print(f"\n\nget_current_temperature.name : \n {get_current_temperature.name}")
print(f"\n\nget_current_temperature.description : \n {get_current_temperature.description}")
print(f"\n\nget_current_temperature.args : \n {get_current_temperature.args}")
# print(f"\n\nget_current_temperature.name : \n {get_current_temperature.name}")

#----------------------------------------------------------------

from langchain.tools.render import format_tool_to_openai_function
get_current_temperature_openai = format_tool_to_openai_function(get_current_temperature)
print(f"\n\nget_current_temperature in Openai Format \n : {get_current_temperature_openai}")

get_current_temperature_openai_2 = get_current_temperature({"latitude": 13, "longitude": 14})
print(f"\n\nget_current_temperature is \n : {get_current_temperature_openai_2}")

#----------------------------------------------------------------

import wikipedia
@tool
def search_wikipedia(query: str) -> str:
    """Run Wikipedia search and get page summaries."""
    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles[: 3]:
        try:
            wiki_page =  wikipedia.page(title=page_title, auto_suggest=False)
            summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
        except (
            self.wiki_client.exceptions.PageError,
            self.wiki_client.exceptions.DisambiguationError,
        ):
            pass
    if not summaries:
        return "No good Wikipedia Search Result was found"
    return "\n\n".join(summaries)

print(f"\n\nsearch_wikipedia.name : \n {search_wikipedia.name}")
print(f"\n\nsearch_wikipedia.description : \n {search_wikipedia.description}")

format_tool_to_openai_function(search_wikipedia)

res_wiki_search = search_wikipedia({"query": "langchain"})
print(f"\n\nWikipedia Search Result : \n : {res_wiki_search}")



#----------------------------------------------------------------

from langchain.chains.openai_functions.openapi import openapi_spec_to_openai_fn
from langchain.utilities.openapi import OpenAPISpec
# from langchain_community.tools import OpenAPISpec

text = """
{
  "openapi": "3.0.0",
  "info": {
    "version": "1.0.0",
    "title": "Swagger Petstore",
    "license": {
      "name": "MIT"
    }
  },
  "servers": [
    {
      "url": "http://petstore.swagger.io/v1"
    }
  ],
  "paths": {
    "/pets": {
      "get": {
        "summary": "List all pets",
        "operationId": "listPets",
        "tags": [
          "pets"
        ],
        "parameters": [
          {
            "name": "limit",
            "in": "query",
            "description": "How many items to return at one time (max 100)",
            "required": false,
            "schema": {
              "type": "integer",
              "maximum": 100,
              "format": "int32"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "A paged array of pets",
            "headers": {
              "x-next": {
                "description": "A link to the next page of responses",
                "schema": {
                  "type": "string"
                }
              }
            },
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Pets"
                }
              }
            }
          },
          "default": {
            "description": "unexpected error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        }
      },
      "post": {
        "summary": "Create a pet",
        "operationId": "createPets",
        "tags": [
          "pets"
        ],
        "responses": {
          "201": {
            "description": "Null response"
          },
          "default": {
            "description": "unexpected error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        }
      }
    },
    "/pets/{petId}": {
      "get": {
        "summary": "Info for a specific pet",
        "operationId": "showPetById",
        "tags": [
          "pets"
        ],
        "parameters": [
          {
            "name": "petId",
            "in": "path",
            "required": true,
            "description": "The id of the pet to retrieve",
            "schema": {
              "type": "string"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Expected response to a valid request",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Pet"
                }
              }
            }
          },
          "default": {
            "description": "unexpected error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        }
      }
    }
  },
  "components": {
    "schemas": {
      "Pet": {
        "type": "object",
        "required": [
          "id",
          "name"
        ],
        "properties": {
          "id": {
            "type": "integer",
            "format": "int64"
          },
          "name": {
            "type": "string"
          },
          "tag": {
            "type": "string"
          }
        }
      },
      "Pets": {
        "type": "array",
        "maxItems": 100,
        "items": {
          "$ref": "#/components/schemas/Pet"
        }
      },
      "Error": {
        "type": "object",
        "required": [
          "code",
          "message"
        ],
        "properties": {
          "code": {
            "type": "integer",
            "format": "int32"
          },
          "message": {
            "type": "string"
          }
        }
      }
    }
  }
}
"""
# import json
# # Parse the OpenAPI spec from a string to a dictionary
# spec_dict = json.loads(text)

# # Initialize the OpenAPISpec from the dictionary
# spec = OpenAPISpec.from_spec_dict(spec_dict)

# Continue with your existing code
# pet_openai_functions, pet_callables = openapi_spec_to_openai_fn(spec)

# spec = OpenAPISpec.from_text(text)

# pet_openai_functions, pet_callables = openapi_spec_to_openai_fn(spec)

# print(f"\n\n pet_openai_functions \n : {pet_openai_functions}")

# #----------------------------------------------------------------

# from langchain.chat_models import ChatOpenAI
# # from langchain_openai import ChatOpenAI
# model = ChatOpenAI(temperature=0).bind(functions=pet_openai_functions)
# resp_3pet = model.invoke("what are three pets names?")
# print(f"\n\n what are three pets names? \n{resp_3pet}")
# resp_petid_42 = model.invoke("tell me about pet with id 42")
# print(f"\n\n tell me about pet with id 42 ? \n{resp_petid_42}")

#----------------------------------------------------------------
# Routing

functions = [
    format_tool_to_openai_function(f) for f in [
        search_wikipedia, get_current_temperature
    ]
]

from langchain_openai import ChatOpenAI
model = ChatOpenAI(temperature=0).bind(functions=functions)

resp_wtr = model.invoke("what is the weather in sf right now")
print(f"\n\n what is the weather in sf right now ? \n {resp_wtr}")
resp_lc = model.invoke("what is langchain")
print(f"\n\n what is langchain? \n {resp_lc}")

#----------------------------------------------------------------

from langchain.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful but sassy assistant"),
    ("user", "{input}"),
])
chain = prompt | model

resp_wtr_2 = chain.invoke({"input": "what is the weather in sf right now"})
print(f"\n\n what is the weather in sf right now? \n {resp_wtr_2}")

#----------------------------------------------------------------

from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

chain = prompt | model | OpenAIFunctionsAgentOutputParser()

result = chain.invoke({"input": "what is the weather in sf right now"})
print(f"type(result) : {type(result)}")
print(f"result.tool : {result.tool}")
print(f"result.tool_input : {result.tool_input}")

get_current_temperature(result.tool_input)
result = chain.invoke({"input": "hi!"})
print(f"type(result) : {type(result)}")
print(f"result.return_values : {result.return_values}")

#----------------------------------------------------------------

from langchain.schema.agent import AgentFinish
def route(result):
    if isinstance(result, AgentFinish):
        return result.return_values['output']
    else:
        tools = {
            "search_wikipedia": search_wikipedia, 
            "get_current_temperature": get_current_temperature,
        }
        return tools[result.tool].run(result.tool_input)
    
chain = prompt | model | OpenAIFunctionsAgentOutputParser() | route

result_2 = chain.invoke({"input": "What is the weather in san francisco right now?"})
print(f"\n\nWhat is the weather in san francisco right now?\n {result_2}")

result_3 = chain.invoke({"input": "What is langchain?"})
print(f"\n\nWhat is langchain?\n {result_3}")

result_4 = chain.invoke({"input": "hi!"})
print(f"\n\nhi!\n {result_4}")

#----------------------------------------------------------------


