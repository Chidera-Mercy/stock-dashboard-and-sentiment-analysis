"""
Web scraping using BeautifulSoup
"""
# import Dependencies
from datetime import datetime
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf

def search_for_stock_news_urls(ticker):
    """
    Search for stock news URLs using a Google search query specific to Yahoo Finance.
    
    Args:
        ticker (str): The stock ticker symbol.
    
    Returns:
        hrefs (list): A list of URLs found in the Google search results.
    """
    # Construct the Google search URL with the stock ticker
    search_url = f"https://www.google.com/search?q=yahoo+finance+{ticker}&tbm=nws"
    req = Request(url=search_url, headers={"user-agent": "my-app"})
    response = urlopen(req)

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response, 'lxml')
    atags = soup.find_all('a')

    # Extract URLs from the anchor tags
    hrefs = [link['href'] for link in atags]
    return hrefs

def strip_unwanted_urls(urls, exclude_list=['maps', 'policies','preferences','accounts', 'support']):
    """
    Filter out unwanted URLs based on an exclusion list.
    
    Args:
        urls (list): A list of URLs to be filtered.
        exclude_list (list): A list of substrings that should be excluded from the URLs.
    
    Returns:
        val (list): A list of cleaned and filtered URLs.
    """
    val = []
    for url in urls:
        if ('https://' in url) and not any(exclude_word in url for exclude_word in exclude_list):
            res = re.findall(r'(https?://\S+)', url)[0].split('&')[0]
            val.append(res)
    return list(set(val))

def fetch_news_data(url):
    """
    Fetch the news headline and publication date from a given URL.
    
    Args:
        url (str): The URL to fetch news data from.
    
    Returns:
        result (list): A list containing the formatted date, headline, and URL, or None if an error occurs.
    """
    try:
        # Send a request to the given URL
        req = Request(url=url, headers={"user-agent": "my-app"})
        response = urlopen(req)
        soup = BeautifulSoup(response, "lxml")

        # Extract the headline from the HTML title tag
        title_tag = soup.find_all('title')
        if not title_tag:
            return None
        headline = title_tag[0].text

        # Check if 'time' tags are present, which contain the publication date
        time_tags = soup.find_all('time')
        if not time_tags:
            return None
        
        # Parse and format the publication date
        date_string = time_tags[0]['datetime']
        dt = datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%S.%fZ")
        formatted_date = dt.strftime("%b-%d-%y %H:%M %S")

        return [formatted_date, headline, url]
    
    except Exception:
        return None

def scrape_and_process(urls, ticker):
    """
    Scrape and process news data from a list of URLs and retrieve corresponding stock prices.
    
    Args:
        urls (list): A list of news URLs to scrape data from.
        ticker (str): The stock ticker symbol.
    
    Returns:
        df (DataFrame): A DataFrame containing the news data (date, headline, and URL).
        stock_df (DataFrame): A DataFrame containing the stock price data (date and price).
    """
    data_array = []
    
    # Use ThreadPoolExecutor to fetch news data concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_news_data, url) for url in urls]
        for future in as_completed(futures):
            result = future.result()
            if result:
                data_array.append(result)
    
    # If no data was collected, return empty DataFrames
    if not data_array:
        return pd.DataFrame(), pd.DataFrame()
    
    # Create a DataFrame from the collected news data
    columns = ['Date Time', 'Headline', 'URL']
    df = pd.DataFrame(data_array, columns=columns)

    try:
        # Convert the 'Date Time' column to a datetime object
        df['Date Time'] = pd.to_datetime(df['Date Time'], format="%b-%d-%y %H:%M %S", utc=True)
    except ValueError:
        df = pd.DataFrame()
    
    # Sort the DataFrame by date and reset the index
    df.set_index('Date Time', inplace=True)
    df.sort_values(by='Date Time', ascending=False)
    df.reset_index(inplace=True)

    try:
        # Retrieve stock prices for the period covered by the news data
        start_date = df['Date Time'].min()
        end_date = df['Date Time'].max()
        stock_data = yf.download(ticker, start=start_date, end=end_date)
    except Exception:
        return df, pd.DataFrame()

    stock_df = stock_data[['Close']].reset_index()
    stock_df.columns = ['Date Time', 'Price']

    try:
        # Convert the 'Date Time' column in the stock data to a datetime object
        stock_df['Date Time'] = pd.to_datetime(stock_df['Date Time'], format="%b-%d-%y %H:%M %S", utc=True)
    except ValueError:
        df = pd.DataFrame()

    return df, stock_df


def output(ticker):
    """
    Orchestrate the entire process of scraping news data, processing it, and retrieving corresponding stock prices.
    
    Args:
        ticker (str): The stock ticker symbol.
    
    Returns:
        df (DataFrame): A DataFrame containing the processed news data.
        stock_df (DataFrame): A DataFrame containing the corresponding stock price data.
    """
    urls = search_for_stock_news_urls(ticker)
    cleaned_urls = strip_unwanted_urls(urls)
    df, stock_df = scrape_and_process(cleaned_urls, ticker)
    return df, stock_df

