import os
from dotenv import load_dotenv
from langchain_community.utilities import BingSearchAPIWrapper
from langchain.tools import Tool
from bs4 import BeautifulSoup
import requests
from datetime import datetime  
import pytz  


load_dotenv()
search = BingSearchAPIWrapper()

# The bingsearch snippet doesn't always provide enough information.
# I am getting the whole webpage content to feed it to the GPT to answer user's questions. 
def BingSearch(query):
    """
    This tool is used to return the WebPage contents and can be used to answer user's questions. 
    """
    headers = {'user-agent':'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.193 Safari/537.36'}
    results = search.results(query, 3) #Number of webpages to check and return content. 
    links = []
    contents = []
    for result in results: 
        #Some pages don't return expected result so we use a try except method to avoid getting an errors. 
        try:
            webpage = requests.get(result['link'], headers)
            soup = BeautifulSoup(webpage.content, 'html.parser')
            text = soup.find('body').get_text().strip()
            cleaned_text = ' '.join(text.split('\n'))
            cleaned_text = ' '.join(text.split())
            contents.append(cleaned_text)
            links.append(result['link'])
        except:
            continue
    return contents, links

def GetTime(timezone):  
    try:  
        # Create a timezone object using pytz  
        tz = pytz.timezone(timezone)  
          
        # Get the current time in that timezone  
        current_time = datetime.now(tz)  
          
        # Format the time to a string if necessary  
        time_str = current_time.strftime('%Y-%m-%d %H:%M:%S %Z%z')  
          
        return time_str  
    except pytz.exceptions.UnknownTimeZoneError:  
        return "Unknown timezone. Please provide a valid timezone."  

def GetWeather(city_name):   

    base_url = "http://api.weatherapi.com/v1/current.json"  
    api_key = os.getenv("Weather_API")
    complete_url = f"{base_url}?key={api_key}&q={city_name}"  
  
    response = requests.get(complete_url)  
  
    # Check if the request was successful  
    if response.status_code == 200:  
        weather_data = response.json()  
        return weather_data  
    else:  
        return "Failed to get weather data."  

def return_tools():
    tool = Tool(
        name="bing_search",
        description="Search Bing for recent results.",
        func=BingSearch
    )

    tool2 = Tool(
        name="check_time",
        description="Used to return country's time",
        func=GetTime
    )

    tool3 = Tool(
        name="check_weather", 
        description="Used to find weather information about a city", 
        func=GetWeather
    )
    return [tool, tool2, tool3]

