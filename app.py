from flask import Flask, request, jsonify, render_template
import yfinance as yf
import pandas as pd
import ta
import joblib
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('model/random_forest_model.pkl')

def fetch_data(ticker, start_date, end_date):
    """Fetch historical data for a given ticker between specified dates."""
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def preprocess_data(data):
    """Preprocess the data by selecting relevant columns and filling missing values."""
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data = data.ffill()  # Forward fill missing values
    return data

def add_features(data):
    """Add technical indicators as features to the data."""
    data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
    data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    data.dropna(inplace=True)  # Drop rows with NaN values
    return data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/data', methods=['GET'])
def get_data():
    ticker = request.args.get('ticker', 'AAPL')
    start_date = request.args.get('start_date', '2020-01-01')
    end_date = request.args.get('end_date', '2023-01-01')

    # Fetch and process the data
    data = fetch_data(ticker, start_date, end_date)
    processed_data = preprocess_data(data)
    features_data = add_features(processed_data)

    # Prepare data for model prediction
    X = features_data[['SMA_50', 'RSI']]
    model_predictions = model.predict(X)

    # Prepare data to send to frontend
    result = {
        "dates": data.index.strftime('%Y-%m-%d').tolist(),
        "close": data['Close'].tolist(),
        "sma_50": features_data['SMA_50'].tolist(),
        "model_predictions": model_predictions.tolist()
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
