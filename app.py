import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import requests
from textblob import TextBlob
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

app = Flask(__name__)
CORS(app)  # Abilita CORS per tutte le route

NEWS_API_KEY = '45125c308ef1479aa9bcadc970888f32'

@app.route('/api/historical_data')
def get_historical_data():
    symbol = request.args.get('symbol', default='AAPL', type=str)
    if not symbol:
        return jsonify({'error': 'No symbol provided'}), 400

    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="max")  # Ottiene tutto lo storico disponibile

        if hist.empty:
            return jsonify({'error': 'No data found for the given symbol'}), 404

        data = []
        for date, row in hist.iterrows():
            data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Open': row['Open'],
                'High': row['High'],
                'Low': row['Low'],
                'Close': row['Close'],
                'Volume': row['Volume']
            })

        print(f"Requested symbol: {symbol}, Returned data count: {len(data)}")
        return jsonify(data)

    except Exception as e:
        print(f"Error fetching data for symbol {symbol}: {e}")
        return jsonify({'error': 'Internal server error'}), 500

def analyze_article_sentiment(article):
    if not article:
        return 'neutral'
    analysis = TextBlob(article)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'

@app.route('/api/sentiment_analysis')
def sentiment_analysis():
    symbol = request.args.get('symbol', default='AAPL', type=str)
    query = f'{symbol} stock'
    url = f'https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}'

    try:
        response = requests.get(url)
        response.raise_for_status()
        news_data = response.json()

        if news_data['status'] != 'ok':
            raise Exception('Error fetching news')

        articles = news_data['articles']
        sentiment_scores = {
            'positive': 0,
            'neutral': 0,
            'negative': 0
        }

        if not articles:
            raise Exception('No articles found')

        for article in articles:
            description = article.get('description')
            title = article.get('title')
            content = description if description else title
            sentiment = analyze_article_sentiment(content)
            sentiment_scores[sentiment] += 1

        return jsonify(sentiment_scores)

    except requests.exceptions.RequestException as e:
        print(f'Error fetching sentiment for {symbol}: {e}')
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        print(f'Error analyzing sentiment for {symbol}: {e}')
        return jsonify({'error': str(e)}), 500

def calculate_technical_indicators(df):
    df['change'] = df['Close'].diff()
    df['sma_50'] = df['Close'].rolling(window=50).mean()
    df['sma_200'] = df['Close'].rolling(window=200).mean()
    df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['rsi'] = calculate_rsi(df['Close'], 14)
    return df

def calculate_rsi(series, period):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

@app.route('/api/train_predict')
def train_predict():
    symbol = request.args.get('symbol', default='AAPL', type=str)
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="5y")

        if hist.empty:
            return jsonify({'error': 'No data found for the given symbol'}), 404

        # Calcola gli indicatori tecnici
        hist = calculate_technical_indicators(hist)

        # Prepara i dati per il modello
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'sma_50', 'sma_200', 'ema_12', 'ema_26', 'macd', 'rsi']
        X = hist[features].dropna()
        y = hist['Close'][X.index]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Modello LSTM
        X_train_reshaped = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test_reshaped = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
        model.add(LSTM(50))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_data=(X_test_reshaped, y_test), verbose=1)

        predictions = model.predict(X_test_reshaped)

        response = {
            'predictions': predictions.flatten().tolist(),
            'actuals': y_test.tolist()
        }

        return jsonify(response)

    except Exception as e:
        print(f"Error during training and prediction: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
