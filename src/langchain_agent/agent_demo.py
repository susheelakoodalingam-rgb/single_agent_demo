import os
from dotenv import load_dotenv

from langchain_openrouter import ChatOpenRouter
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_agent
from langchain.tools import tool
import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry
import streamlit as st

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPEN_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

######### STREAMLIT PAGE ###################

st.set_page_config(
    page_title="Agentic AI Assistant",
    page_icon="👋",
    layout="centered"
)

st.title("👋 Agentic AI Assistant")
st.markdown("Search + Weather AI Agent Using Langchain")


# Tavily search tool
search_tool = TavilySearchResults(
    max_results=3
)
@tool
def get_weather_data(city:str)->str:
    """
    Fetch current weather information for a city
    """
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
	    "latitude": 52.52,
	    "longitude": 13.41,
	    "hourly": "temperature_2m",
    }
    responses = openmeteo.weather_api(url, params = params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation: {response.Elevation()} m asl")
    print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()

    hourly_data = {
        "date": pd.date_range(
            start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
            end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
            freq = pd.Timedelta(seconds = hourly.Interval()),
            inclusive = "left"
        )
    }

    hourly_data["temperature_2m"] = hourly_temperature_2m

    hourly_dataframe = pd.DataFrame(data = hourly_data)
    return hourly_dataframe

# OpenRouter LLM
llm = ChatOpenRouter(
    model="qwen/qwen3.8-27b",
    temperature=0,
    api_key=OPENROUTER_API_KEY,
    max_tokens=3000
)

system_prompt = """
You are an intelligent AI agent.

You have access to a web search tool.

When answering a user:
1. Understand the user's objective.
2. Determine whether web search is required.
3. Use the search tool when current or external information is required.
4. Use the tool results to answer accurately.
5. If the user asks multiple questions, complete all parts.
6. Do not claim that you searched the web unless you actually used the search tool.
7. Provide a concise final answer.
"""

tools = [search_tool, get_weather_data]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
    
)





user_query=st.text_input("Enter your query:", placeholder="Example: Find the capital of India and current weather ")

if st.button("Run Agent"):
    if user_query:
        with st.spinner("Agent is thinking..."):
            try:
                response = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": user_query
                            }
                        ]
                    }
                )
                st.success("Response Generated")
                st.markdown("## Final Response")
                st.write(response["messages"][-1].content)

            except Exception as e:
                st.error(f"Error: {str(e)}")    