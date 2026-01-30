# Time Series Forecast with Prophet

## Overview

This project performs a 30-day time series forecast using **Facebook Prophet**. It loads historical data from a CSV file, trains a forecasting model with yearly and weekly seasonality, includes Brazilian holidays, and visualizes the results.

## Tech Stack

* Python 3
* pandas
* prophet
* matplotlib

## Data Requirements

The input file must be a CSV named `DBreceipt.csv` with the following columns:

* `data`: date in `DD/MM/YYYY` format
* `analises`: numeric values to be forecast

The file must use `;` as the column separator.

## How It Works

1. Load and preprocess the dataset
2. Rename columns to Prophet format (`ds`, `y`)
3. Train a Prophet model with seasonality and BR holidays
4. Generate a 30-day future forecast
5. Plot forecast results and model components

## Forecast Output

* Main forecast plot (next 30 days)
* Trend, weekly, and yearly seasonality components

## Notes

* Forecast horizon is fixed at 30 days
* Country holidays are set to Brazil (`BR`)
* Plots are displayed using matplotlib
