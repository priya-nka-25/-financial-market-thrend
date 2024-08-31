import yfinance as yf
import ta
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Fetch historical stock data
ticker = 'AAPL'
start_date = '2020-01-01'
end_date = '2023-01-01'

data = yf.download(ticker, start=start_date, end=end_date)

# Preprocess the data
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].ffill()

# Add technical indicators
data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
data.dropna(inplace=True)

# Define features and target
X = data[['SMA_50', 'RSI']]
y = data['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'model/random_forest_model.pkl')

print("Model training complete and saved as 'random_forest_model.pkl'.")
