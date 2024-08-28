"""
Project: Stock Dasboard using Streanlit
"""

# Import Dependencies
from datetime import date, timedelta
import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import numpy as np
import pandas as pd
import plotly.express as px
import scrape_and_summarize
from sentiment.FinbertSentiment import FinbertSentiment

# Constants for stock forecasting
START_FORECAST = "2015-01-01"
END_FORECAST = date.today().strftime("%Y-%m-%d")

@st.cache_data
def load_data(tckr, start, end):
    """
    Load stock data for a given ticker and date range from Yahoo Finance.

    Args:
        ticker (str): Stock ticker symbol.
        start (str): Start date for the data in 'YYYY-MM-DD' format.
        end (str): End date for the data in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame containing stock data with reset index.
    """
    try:
        df = yf.download(tckr, start=start, end=end)
        df.reset_index(inplace=True)
        if df.empty:
            raise ValueError("No data returned from Yahoo Finance.")
        return df
    except Exception:
        return pd.DataFrame()  # Return an empty DataFrame on error

def plot_raw_data(df):
    """
    Plot time series data for stock open and close prices.

    Args:
        df (pd.DataFrame): DataFrame containing stock data with 'Date', 'Open', and 'Close' columns.

    Returns:
        None: The function displays the plot using Streamlit.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"], 
        y=df["Open"], 
        name="stock_open",
        line=dict(color="green")))

    fig.add_trace(go.Scatter(
        x=df["Date"], 
        y=df["Close"], 
        name="stock_close", 
        line=dict(color="blue")))

    fig.layout.update(
        title_text="Time Series Data",
        xaxis_rangeslider_visible=True)
    
    st.plotly_chart(fig)

def score_news(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply sentiment analysis on news articles and calculate sentiment scores.

    Args:
        news_df (pd.DataFrame): DataFrame containing news data with headlines and dates.

    Returns:
        pd.DataFrame: DataFrame with sentiment scores appended.
    """
    sentimentAlgo.set_data(news_df)
    sentimentAlgo.calc_sentiment_score()
    return sentimentAlgo.df


def plot_sentiment(df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Plot sentiment analysis results for a given stock.

    Args:
        df (pd.DataFrame): DataFrame containing sentiment scores.
        ticker (str): Stock ticker symbol.

    Returns:
        go.Figure: Plotly figure object displaying sentiment analysis.
    """
    return sentimentAlgo.plot_sentiment()

def plot_hourly_price(df, ticker) -> go.Figure:
    """
    Plot hourly price data for a given stock.

    Args:
        df (pd.DataFrame): DataFrame containing hourly price data with 'Date Time' and 'Price' columns.
        ticker (str): Stock ticker symbol.

    Returns:
        go.Figure: Plotly figure object displaying hourly price data.
    """
    fig = px.line(
        data_frame=df, 
        x=df['Date Time'],
        y="Price", 
        title=f"{ticker} Price")
    return fig

def display_news_as_text(df):
    """
    Display news articles with sentiment analysis results as text.

    Args:
        df (pd.DataFrame): DataFrame containing news headlines, sentiment scores, and URLs.

    Returns:
        None: The function displays the news articles in Streamlit.
    """
    for index, row in df.iterrows():
        st.write(f"**Date:** {row['Date Time']}")
        st.write(f"**Headline:** {row['Headline']}")
        st.write(f"**Sentiment:** {row['sentiment'][0]['label']} (Score: {row['sentiment_score']:.2f})")
        st.write(f"[Read more]({row['URL']})")
        st.write("---")  # Separator between entries

# Application Title
st.title("Stock Dashboard and Trend Forecast")

# Sidebar configuration for user input
ticker = st.sidebar.text_input('Ticker', value='GS')
start_date = st.sidebar.date_input("Start Date", value=date.today() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date")
n_years = st.sidebar.slider("Years of prediction:", 1, 4)
period = n_years * 365

# Load data and plot stock prices
data = load_data(ticker, start_date, end_date)
if data.empty:
    st.write("Unable to retrieve data.")
else:
    plot_raw_data(data)

# Load data for stock forecasting
forecast_data = load_data(ticker, START_FORECAST, END_FORECAST)

# Define tabs for different features
help_tab, pricing_data, financials, forecast, news,  = st.tabs([
  "Help",
  "Pricing",
  "Financials",
  "Forecast",
  "News and Sentiment Analysis"
])

# Help tab content
with help_tab:
    st.title("Help and Tutorial")
    st.write("""
        **Welcome to the Stock Dashboard and Trend Forecast!**
        
        This application allows you to visualize stock price movements, review key financial statements, forecast future trends and analyze sentiment from news headlines. Here's how to use the features:
        
        - **Ticker**: Enter the stock ticker symbol (e.g., AAPL for Apple) to fetch data.
        - **Start Date / End Date**: Select the date range for historical stock price data.
        - **Years of Prediction**: Adjust the slider to choose how many years into the future you want to forecast.
        - **Pricing Data**: View historical stock prices, volume, and calculate key metrics like annual return and risk-adjusted return.
        - **Financials**: Review the company's income statement, balance sheet, and cash flow statement to assess financial health.
        - **Forecast**: See future price predictions and their components.
        - **News and Sentiment Analysis**: Read recent news headlines and view sentiment analysis alongside stock prices.
        
        For any issues or questions, please contact support.
    """)

# Pricing data tab content
with pricing_data:
    st.header("Price Movements")
    st.subheader(f"{ticker} data")

    if data.empty:
        st.write("Unable to retrieve data.")
    else:
        data2 = data.copy()
        data2['% Change'] = data['Adj Close'] / data['Adj Close'].shift(1) - 1
        data2.dropna(inplace=True)
        st.write(data2)

        annual_return = data2['% Change'].mean() * 252 * 100
        st.subheader(f'Annual Return: {annual_return:.3f}%')

        stdev = np.std(data2['% Change']) * np.sqrt(252)
        st.subheader(f'Standard Deviation: {stdev * 100:.3f}%')

        st.subheader(f'Risk Adj. Return:    {annual_return / (stdev * 100):.3f}')

# Financials tab content
with financials:
    st.header(f"{ticker} Financial Statements and Visualizations")

    # Fetch financial data using yfinance
    ticker_obj = yf.Ticker(ticker)

    # Display Income Statement
    st.subheader("Income Statement")
    income_statement = ticker_obj.financials.T
    if not income_statement.empty:
        st.dataframe(income_statement)

        # Income Statement Visualization - Revenue and Net Income Trends
        st.subheader("Revenue and Net Income Trends")
        fig_income = go.Figure()

        fig_income.add_trace(go.Scatter(
            x=income_statement.index, 
            y=income_statement['Total Revenue'], 
            mode='lines', 
            name='Revenue', 
            line=dict(color='blue')))
        fig_income.add_trace(go.Scatter(
            x=income_statement.index, 
            y=income_statement['Net Income'], 
            mode='lines', 
            name='Net Income', 
            line=dict(color='red')))
        fig_income.update_layout(
            title="Revenue vs Net Income", 
            xaxis_title="Date", 
            yaxis_title="Amount", 
            template="plotly_dark")
        st.plotly_chart(fig_income)
    else:
        st.write("Income statement unavailable")


    # Display Balance Sheet
    st.subheader("Balance Sheet")
    balance_sheet = ticker_obj.balance_sheet.T
    if not balance_sheet.empty:
        st.dataframe(balance_sheet)

        # Balance Sheet Visualization - Asset vs Liabilities
        st.subheader("Assets vs Liabilities")
        fig_balance = go.Figure()

        fig_balance.add_trace(go.Bar(
            x=balance_sheet.index, 
            y=balance_sheet['Total Assets'], 
            name='Total Assets', 
            marker_color='blue'))
        fig_balance.add_trace(go.Bar(
            x=balance_sheet.index, 
            y=balance_sheet['Total Liabilities Net Minority Interest'],
            name='Total Liabilities', 
            marker_color='green'))
        fig_balance.update_layout(
            barmode='group', 
            title="Assets vs Liabilities", 
            xaxis_title="Date", 
            yaxis_title="Amount", 
            template="plotly_dark")
        st.plotly_chart(fig_balance)
    else:
        st.write("Balance sheet unavailable")


    # Display Cash Flow Statement
    st.subheader("Cash Flow Statement")
    cash_flow = ticker_obj.cashflow.T
    if not cash_flow.empty:
        st.dataframe(cash_flow)

        # Cash Flow Statement Visualization - Cash Flows Breakdown
        st.subheader("Cash Flows Breakdown")
        fig_cashflow = go.Figure()

        fig_cashflow.add_trace(go.Bar(
            x=cash_flow.index, 
            y=cash_flow['Cash Flow From Continuing Operating Activities'], 
            name='Operating Activities', 
            marker_color='blue'))
        fig_cashflow.add_trace(go.Bar(
            x=cash_flow.index, 
            y=cash_flow['Cash Flow From Continuing Investing Activities'], 
            name='Investing Activities', 
            marker_color='red'))
        fig_cashflow.add_trace(go.Bar(
            x=cash_flow.index, 
            y=cash_flow['Cash Flow From Continuing Financing Activities'], 
            name='Financing Activities', 
            marker_color='green'))
        fig_cashflow.update_layout(
            barmode='stack', 
            title="Cash Flows Breakdown", 
            xaxis_title="Date", yaxis_title="Amount", 
            template="plotly_dark")
        st.plotly_chart(fig_cashflow)
    else:
        st.write("Cash flow unavailable")

# Forecast tab content
with forecast:
    # Check if forecast data is available and not empty
    if forecast_data is not None and not forecast_data.empty:
        # Prepare the training data for the forecasting model (Prophet)
        df_train = forecast_data[["Date", "Close"]]
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

        # Initialize and train the Prophet model
        m = Prophet()
        m.fit(df_train)

        # Create future dates for the forecasting period
        future = m.make_future_dataframe(periods=period)
        # Predict future stock prices
        forecast = m.predict(future)

        # Display forecast data
        st.subheader("Forecast Data")
        st.write(forecast)

        # Plot and display the forecast data
        st.write("Forecast Data")
        plot1 = plot_plotly(m, forecast)
        st.plotly_chart(plot1)

        # Plot and display the forecast components (trend, weekly, yearly)
        st.write("Forecast Components")
        plot2 = m.plot_components(forecast)
        st.write(plot2)
    else:
        # Handle cases where forecast data could not be retrieved
        st.write("Unable to retrieve data.")

# Initialize FinBERT sentiment analysis model
sentimentAlgo = FinbertSentiment()
sentimentAlgo.set_symbol(ticker)

# News tab content
with news:
    st.header("Stock News Summary and Analysis")
    temp_text = st.text("Please wait, gathering news and performing analysis...")

    # Scrape and summarize news data along with stock price data
    news_df, price_df = scrape_and_summarize.output(ticker)

    # Check if news or price data is available and not empty
    if news_df.empty or price_df.empty:
        st.write("Unable to retrieve data.")
    else:
        # Perform sentiment analysis on the gathered news data
        scored_news_df = score_news(news_df)

        # Plot and display the sentiment analysis results
        fig_bar_sentiment = plot_sentiment(scored_news_df, ticker)
        st.plotly_chart(fig_bar_sentiment)

        # Plot and display the historical price data 
        fig_line_price_history = plot_hourly_price(price_df, ticker)
        st.plotly_chart(fig_line_price_history)

        # Display the news headlines along with sentiment scores
        st.subheader('News Headlines')
        display_news_as_text(scored_news_df)
        
    # Update the temporary message to indicate completion of the analysis
    temp_text.text("Analysis complete! Here are the results:")
