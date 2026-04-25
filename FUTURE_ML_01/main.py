import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load dataset
data = pd.read_csv("sales_data.csv")

# Convert date column
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Show first few rows
print(data.head())

# Plot sales data
plt.figure(figsize=(10,5))
plt.plot(data['Sales'])
plt.title("Sales Over Time")
plt.show()

# Train ARIMA model
model = ARIMA(data['Sales'], order=(5,1,0))
model_fit = model.fit()

# Forecast future values
forecast = model_fit.forecast(steps=10)

print("\nFuture Predictions: \n", forecast)
print(forecast)

# Plot forecast
plt.figure(figsize=(10,5))
plt.plot(data['Sales'], label='Actual')
plt.plot(forecast, label='Forecast')
plt.legend()
plt.show()