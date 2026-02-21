import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from src.time_series import load_cleaned_data, aggregate_daily_price

df = load_cleaned_data()
daily = aggregate_daily_price(df)
y = daily["avg_price"]
print(daily.columns)

train = y[:25]
test = y[25:]

model = ARIMA(train, order=(1,1,1))
fit = model.fit()

forecast = fit.forecast(steps=len(test))

plt.figure(figsize=(10,5))
plt.plot(train.index, train, label="Train")
plt.plot(test.index, test, label="Actual", marker='o')
plt.plot(test.index, forecast, label="Forecast", marker='o')
plt.legend()
plt.title("ARIMA Forecast vs Actual")
plt.show()

from sklearn.metrics import mean_squared_error
import numpy as np

rmse = np.sqrt(mean_squared_error(test, forecast))
print("ARIMA RMSE:", rmse)


# ===== Residuals =====
residuals = test - forecast

plt.figure(figsize=(8,4))
plt.plot(residuals)
plt.title("Residuals Over Time")
plt.show()

plt.figure(figsize=(6,4))
plt.hist(residuals, bins=10)
plt.title("Residual Distribution")
plt.show()

