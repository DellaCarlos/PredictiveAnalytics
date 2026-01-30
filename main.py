import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os

# loading data
df = pd.read_csv('DBreceipt.csv', sep=';')
df = df.rename(columns={'data': 'ds', 'analises': 'y'})
df['ds'] = pd.to_datetime(df['ds'], format='%d/%m/%Y')

# Training
model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
model.add_country_holidays(country_name='BR')
model.fit(df)

future = model.make_future_dataframe(periods=30) # 30-day forecast
forecast = model.predict(future)

fig = model.plot(forecast)
ax = fig.gca()
ax.set_title("Estimated Delivery Date – Next 30 Days", fontsize=12, pad=15)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Delivery", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.4)
ax.tick_params(axis="both", labelsize=10)
plt.tight_layout()
plt.show()

flg2 = model.plot_components(forecast)
plt.show()