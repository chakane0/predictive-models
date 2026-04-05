# Stock Analysis

The fullpipeline would include: (1) Earning call audio, (2) Whisper - for text segments, (3) FinBERT - sentiment analysis, (4) Sentiment score, (5) Historical stock data, (6) Feature engineering, (7) Combined features, (8) model, (9) Prediction, (10) Market Indicators, (11) volume data, (12) up or down the next day.

On a high level the model will predict wether a stock will be up or down after an earnings call. The features of this model inclues historical price movement, volume changes, sentiment score, market conditions.



### Notes

A fundamental way of turning raw price data into features models can learn from common time series features include:

1. Daily(intra return) return (percent change from yesterday)
    a. Measures the percentage gain/loss of an investment in a single trading day. This monitors short term portfolio volatility and performance. 

2. Volatility (how much prices bounce around)
3. Moving Averages (trend indicators)
    a. A calculation to analyze data points by creating a series of averages of different selections of the full dataset. The variations include, simple, cumlative, or weighted forms
4. Volume changes




