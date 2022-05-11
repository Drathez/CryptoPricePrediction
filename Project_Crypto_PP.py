# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
 # """
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import csv   

from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping

# Load Data
crypto_currency = 'BTC'
against_currency = 'USD'


f'{crypto_currency}'
start = dt.datetime(2014, 9, 17)
end = dt.datetime(2021, 1, 1)

data = web.DataReader(f'{crypto_currency}-{against_currency}', 'yahoo', start, end)

# Prepare Data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

prediction_days = 30

x_train = [] 
y_train = []

for x in range(prediction_days, len(scaled_data)):      
    x_train.append(scaled_data[x-prediction_days:x, 0])   
    y_train.append(scaled_data[x, 0])
    
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Load Test Data
test_start = dt.datetime(2021, 1, 1)
test_end = dt.datetime.now()

test_data = web.DataReader(f'{crypto_currency}-{against_currency}', 'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

total_date =[]

for x in range(0, len(test_data)):  
    total_date.append(test_data.index[x])

total_date = np.array(total_date)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

x_test = []
y_test = []

for x in range(prediction_days, len(model_inputs)):
   x_test.append(model_inputs[x-prediction_days:x, 0])
   y_test.append(model_inputs[x, 0])
    
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

''' Test The Model Accuracy on Existing Data '''

# Build The Model
model = Sequential()

# This callback will stop the training when there is no improvement in  
# the loss for four consecutive epochs. 
callback = EarlyStopping(monitor='loss', patience=4)

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1)) # Prediction of the next closing price

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae', 'mape'])

lstmModel = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=32, callbacks=[callback])

# Make Prediction on Test Data
predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Get today's predicted price
prediction_today = predicted_prices[len(predicted_prices)-1][0]

              
# Predict Next Day (+1 of current date)
last_day = total_date[-1] + dt.timedelta(days=1)

real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs) + 1, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction_tomorrow = model.predict(real_data)
prediction_tomorrow = scaler.inverse_transform(prediction_tomorrow)

  
# Plot Bitcoin Price Prediction Chart
fig, ax = plt.subplots()

# Label in the x axis ticks will be displayed in 'mm-YYYY' format.
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))

# Rotate and positions the x axis ticks for better alignment and views
for label in ax.get_xticklabels(which='major'):
    label.set(rotation=30, horizontalalignment='right')
    
plt.plot(total_date, actual_prices, color="Black", label=f"Actual {crypto_currency}-{against_currency} Price")
plt.plot(total_date, predicted_prices, color="Green", label=f"Predicted {crypto_currency}-{against_currency} Price" )
plt.plot(last_day, prediction_today, 'bo', color="Blue", label="Today's Price Prediction")

plt.title(f"{crypto_currency}-{against_currency} Price ($)")
plt.xlabel('Date')
plt.ylabel(f"{crypto_currency}-{against_currency} Price ($)")
plt.legend()
plt.savefig(r"C:\CryptoPPProject\PythonData\BTCPP.png", bbox_inches ="tight", format="png", dpi=299)
plt.show()

# Plot Training & Test History
plt.plot(lstmModel.history['mse'], label='train')
plt.plot(lstmModel.history['val_mse'], label='test')
plt.xlabel('Number of Epochs')
plt.ylabel('Value of MSE')
plt.legend()
plt.show()

#######################################################################################################################

# Twitter Sentiment Analysis for Crypto Price Prediction Dashboard

import re
import tweepy
from textblob import TextBlob

# Function to clean tweet text by removing links, special characters, using simple regex statements
def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

consumer_key = #your consumer key
consumer_secret = #your consumer secret key
access_token = #your access token
access_token_secret = #your access secret token

auth_handler = tweepy.OAuthHandler(consumer_key = consumer_key, consumer_secret = consumer_secret)
auth_handler.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth_handler, wait_on_rate_limit=True)

search_term = 'bitcoin'
tweet_amount = 1000
since_date = dt.date.today() - dt.timedelta(4)  # Get date of 4 days ago
until_date = dt.date.today() - dt.timedelta(3)  # Get date of 3 days ago

tweets = tweepy.Cursor(api.search, q=search_term, since=since_date, until=until_date, lang='en').items(tweet_amount)

ref_tweets = tweet_amount / 1924891  # As referenced in article
cross_correlation = 0.35
total_correlation = ref_tweets * cross_correlation

polarity = 0

positive = 0
neutral = 0
negative = 0

for tweet in tweets:
    final_text = tweet.text.replace('RT', '')
    if final_text.startswith(' @'):
        position = final_text.index(':')
        final_text = final_text[position+2:]
    if final_text.startswith('@'):
        position = final_text.index(' ')
        final_text = final_text[position+2:]
    analysis = TextBlob(clean_tweet(final_text))
    tweet_polarity = analysis.polarity   
    if tweet_polarity > 0:
        positive += 1
    elif tweet_polarity < 0:
        negative += 1
    else:
        neutral += 1
    polarity += analysis.polarity
    
    
polarity = polarity/tweet_amount * 1
percentage_positive = (positive / tweet_amount) * 100
percentage_neutral = (neutral / tweet_amount) * 100
percentage_negative = (negative / tweet_amount) * 100

sentiment_PP_today = round(float(prediction_today * (1-total_correlation)), 2)

sentiment_PP_tomorrow = round(float(prediction_tomorrow * (1-total_correlation)), 2)

print(polarity)
print(f"Amount of positive tweets: {positive} ({percentage_positive}%)")
print(f"Amount of negative tweets: {negative} ({percentage_negative}%)")
print(f"Amount of neutral tweets: {neutral} ({percentage_neutral}%)")
print(f"Total tweets: {tweet_amount}")

print("Before Sentiment Analysis (Today): ", round(float(prediction_today),2))
print("After Sentiment Analysis (Today): ", sentiment_PP_today)

# Export daily BTC price data to .csv format, to be used for dashboard
# and update yesterday's closing price and prediction accuracy for dashboard

Latest_BTC_Price = round(float(test_data.loc[test_data.index[len(test_data)-1], "Close"]), 2)

Prediction_Accuracy = round(float(100 - (abs(sentiment_PP_today - Latest_BTC_Price)/Latest_BTC_Price * 100)), 2)

# Standardize date format
today_date = dt.datetime.strptime(str(dt.date.today()), '%Y-%m-%d').strftime('%d/%m/%Y')
yesterday_date = dt.datetime.strptime(str(dt.date.today() - timedelta(days = 1)), '%Y-%m-%d').strftime('%d/%m/%Y')

fields=[today_date, "", sentiment_PP_today, sentiment_PP_tomorrow, ""]

with open(r'C:\CryptoPPProject\PythonData\CryptoPPData.csv', 'a', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(fields)
    
with open(r'C:\CryptoPPProject\PythonData\CryptoPPData.csv', 'r') as f:
    r = csv.reader(f)
    datacsv = [line for line in r]

for x in range(1, len(datacsv)):
    if dt.datetime.strptime(datacsv[x][0], '%d/%m/%Y')  == dt.datetime.strptime(yesterday_date, '%d/%m/%Y'):
        datacsv[x][1] = round(actual_prices[len(actual_prices)-1], 2)
        datacsv[x][4] = round(float(100 - ((abs(float(datacsv[x][2]) - float(datacsv[x][1]))/float(datacsv[x][1])) * 100)), 2)
        break

with open(r'C:\CryptoPPProject\PythonData\CryptoPPData.csv', 'w', newline="") as f:
    w = csv.writer(f)
    w.writerows(datacsv)
