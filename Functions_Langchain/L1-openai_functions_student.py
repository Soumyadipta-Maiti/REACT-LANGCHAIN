import langchain_openai
import os
import openai
from dotenv import load_dotenv
load_dotenv()

import json

def get_current_weather(location: str, unit: str = "fahrenheit") -> json:
    """Get the current weather in the given location"""
    weather_info = {
        "location": location,
        "unit": unit,
        "temperature": 72,
        "forcast": ["sunny", "windy"]        
    }
    
    return json.dumps(weather_info)

# # print(get_current_weather("kolkata"))
# weather_json = json.loads(get_current_weather("kolkata"))
# print(weather_json["temperature"])

functions = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in the given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "fahrenheit"}
        },
            "required": ["location"],
        },
    }
]

messages = [
    {
        "role": "user",
        "content": "what is the weather in Kolkata?"
    }
]


from openai import OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model = "gpt-4",
    # model = "gpt-3.5-turbo-0613",
    messages = messages,
    functions = functions
)

print(response)


arguments_str = response.choices[0].message.function_call.arguments


args = json.loads(arguments_str)

print(get_current_weather(args))

#----------------------------------------------------------------

messages = [
    {
        "role": "user",
        "content": "hi",
    }
]

response_1 = client.chat.completions.create(
    model = "gpt-4",
    messages = messages,
    functions = functions
)

print(response_1)

#----------------------------------------------------------------

response_2 = client.chat.completions.create(
    model = "gpt-4",
    messages = messages,
    functions = functions,
    function_call="auto",
)

print(response_2)

#----------------------------------------------------------------


response_3 = client.chat.completions.create(
    model = "gpt-4",
    messages = messages,
    functions = functions,
    function_call="none",
)

print(response_3)

#----------------------------------------------------------------

messages = [
    {
        "role": "user",
        "content": "What's the weather in Boston?",
    }
]
response_4 = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    functions=functions,
    function_call="none",
)
print(response_4)

#----------------------------------------------------------------

messages = [
    {
        "role": "user",
        "content": "hi",
    }
]

response_5 = client.chat.completions.create(
    model = "gpt-4",
    messages = messages,
    functions = functions,
    function_call={"name": "get_current_weather"},
)

print(response_5)

#----------------------------------------------------------------

messages = [
    {
        "role": "user",
        "content": "What's the weather like in Boston!",
    }
]

response_5 = client.chat.completions.create(
    model = "gpt-4",
    messages = messages,
    functions = functions,
    function_call={"name": "get_current_weather"},
)

print(response_5)

#----------------------------------------------------------------

messages.append(response_5.choices[0].message)

args = json.loads(response_5.choices[0].message.function_call.arguments)
observation = get_current_weather(args)

messages.append(
        {
            "role": "function",
            "name": "get_current_weather",
            "content": observation,
        }
)

response_6 = client.chat.completions.create(
    model = "gpt-4",
    messages = messages,
    # functions = functions,
    # function_call={"name": "get_current_weather"},
)

print(response_6)

#----------------------------------------------------------------

tool_functions = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in the given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "fahrenheit"}
            },
                "required": ["location"],
                "additionalProperties": False,
            },
            }
        }
]

# tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "get_delivery_date",
#             "description": "Get the delivery date for a customer's order. Call this whenever you need to know the delivery date, for example when a customer asks 'Where is my package'",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "order_id": {
#                         "type": "string",
#                         "description": "The customer's order ID.",
#                     },
#                 },
#                 "required": ["order_id"],
#                 "additionalProperties": False,
#             },
#         }
#     }
# ]

messages = [
    {
        "role": "user",
        "content": "what is the weather in Kolkata?"
    }
]


resp_11 = openai.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tool_functions,
    )

print(f"\n \n FUNCTION CALLING USING LATEST TOOLS : \n \n {resp_11}")

# arguments_str_11 = resp_11.choices[0].message.function_call.arguments
# args_11 = json.loads(arguments_str_11)


tool_call = resp_11.choices[0].message.tool_calls[0]
args_11 = json.loads(tool_call.function.arguments)
# args_11 = json.loads(tool_call['function']['arguments'])


print(get_current_weather(args_11))

#----------------------------------------------------------------

messages_hi = [
    {
        "role": "user",
        "content": "Hi !"
    }
]


resp_111 = openai.chat.completions.create(
    model="gpt-4o",
    messages=messages_hi,
    tools=tool_functions,
    )


# args_111 = resp_111.choices[0].message.tool_calls[0].function.arguments
args_111 = resp_111.choices[0].message.content
args_111 = json.loads(args_111)
print(args_111)
print(get_current_weather(args_111))


    