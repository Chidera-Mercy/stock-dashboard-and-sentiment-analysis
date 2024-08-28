"""
Utilize FinBERT for sentiment analysis
"""
# Import Dependencies
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

@st.cache_resource
def load_finbert_model():
    """
    Loads and caches the FinBERT model and tokenizer.
    """
    model_path = "FinbertModel"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

class SentimentAnalysisBase():
    """
    A base class for sentiment analysis, providing methods to set data,
    calculate sentiment scores, and plot sentiment analysis results.

    Attributes:
        symbol (str): The stock symbol for which sentiment analysis is performed.
        df (pd.DataFrame): The DataFrame containing news headlines and sentiment scores.
    """

    def __init__(self):
        pass

    def set_symbol(self, symbol):
        """
        Sets the stock symbol for sentiment analysis.

        Parameters:
            symbol (str): The stock symbol to be analyzed.
        """
        self.symbol = symbol

    def set_data(self, df):
        """
        Sets the DataFrame containing news headlines for sentiment analysis.

        Parameters:
            df (pd.DataFrame): The DataFrame with news headlines and associated data.
        """
        self.df = df

    def calc_sentiment_score(self):
        pass

    def get_sentiment_scores(self):
        """
        Retrieves the DataFrame with sentiment scores.

        Returns:
            pd.DataFrame: The DataFrame containing sentiment scores.
        """
        return self.df

    def plot_sentiment(self) -> go.Figure:
        """
        Plots sentiment scores as a bar chart.

        Returns:
            go.Figure: A Plotly Figure object representing the sentiment scores.
        """

        # Define color scheme for sentiment labels
        colors = {
            'positive': 'green',
            'negative': 'red',
            'neutral': 'gray'
        }

        # Remove rows with zero sentiment score
        df_plot = self.df.drop(self.df[self.df['sentiment_score'] == 0].index)

        # Create a bar plot of sentiment scores
        fig = px.bar(
            data_frame=df_plot, 
            x=df_plot['Date Time'], 
            y='sentiment_score',
            color = df_plot['sentiment'].apply(lambda x: x[0]['label']),
            color_discrete_map={
                'positive': colors['positive'],
                'negative': colors['negative'],
                'neutral': colors['neutral']},
            title=f"{self.symbol} Sentiment Scores",
            labels={'sentiment_score': 'Sentiment Score', 'Date Time': 'Date Time'})
        
        # Update hover template for better display
        fig.update_traces(
            hovertemplate="<br>".join([
                "Date: %{x}",
                "Score: %{y:.2f}",
                "<extra></extra>"  # Removes extra info
            ])
        )
        
        # Update layout with axis titles
        fig.update_layout(
            xaxis_title="Date Time",
            yaxis_title="Sentiment Score",
            title=f"{self.symbol} Sentiment Analysis"
        )
        return fig


class FinbertSentiment (SentimentAnalysisBase):
    """
    A subclass of SentimentAnalysisBase that uses the FinBERT model for sentiment analysis.
    
    Attributes:
        _sentiment_analysis (pipeline): The sentiment analysis pipeline using FinBERT.
    """
    def __init__(self):
        """
        Initializes the FinbertSentiment class and loads the FinBERT model and tokenizer.
        """
        self._sentiment_analysis = load_finbert_model()
        super().__init__()

    def calc_sentiment_score(self):
        """
        Calculates sentiment scores for the data using the FinBERT model.
        Updates the DataFrame with sentiment labels and scores.
        """
        # Apply sentiment analysis to the headlines
        self.df['sentiment'] = self.df['Headline'].apply(
            self._sentiment_analysis)
        
        # Calculate sentiment scores based on labels and scores
        self.df['sentiment_score'] = self.df['sentiment'].apply(
            lambda x: {x[0]['label'] == 'negative': -1, x[0]['label'] == 'positive': 1}.get(True, 0) * x[0]['score'])
