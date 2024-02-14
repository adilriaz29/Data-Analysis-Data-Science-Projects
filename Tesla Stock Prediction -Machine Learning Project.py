#!/usr/bin/env python
# coding: utf-8

# # Data Project - Stock Market Analysis 📈
# 
# 

# Within this notebook, our focus lies squarely on the performance analysis of Tesla stock, a pivotal player in the technology and automotive sectors. Through the lens of time series data, we aim to unravel the intricate patterns and trends embedded within Tesla's historical stock performance, leveraging Python libraries such as yfinance, Seaborn, Matplotlib, and LSTM.
# 
# Our journey commences with the acquisition of Tesla stock data. This robust Python tool enables seamless retrieval of market data, laying the groundwork for our analysis.
# 
# Throughout our exploration, we set out to address key questions:
# 
# > ### What is the average daily return of Tesla stock?
# > ### How do moving averages shed light on Tesla's stock trends?
# > ### What level of risk is associated with investing in Tesla stock?
# 
# Can LSTM models offer insights into future stock behavior, exemplified by predicting Tesla's closing price?
# By delving into these inquiries, we endeavor to deepen our understanding of Tesla's market dynamics, thereby sharpening our analytical acumen in the realm of data science.
# 
# 

# # Loading Libraries & Data Exploration 📶

# In[353]:


import os
import pandas as pd
import numpy as np
import math
import datetime as dt
import matplotlib.pyplot as plt


from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping


import warnings
warnings.filterwarnings("ignore")
import plotly.express as px

from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# In[354]:


df = pd.read_csv('TSLA.csv')
df.head(10)


# In[355]:


df.tail(10)


# In[356]:


# Summary Stats
df.describe()


# In[357]:


# General info
df.info()


# In[358]:


print("Total number of days: ",df.shape[0])
print("Total number of fields: ",df.shape[1])


# In[359]:


print("Null values:", df.isnull().values.sum())
print("NA values:", df.isna().values.any())


# In[391]:


df["Date"]=pd.to_datetime(df["Date"])


# In[392]:


tesla_spec=df[["Date","Close"]]


# In[393]:


tesla_spec.head(15)


# In[394]:


print("Starting date: ",df.iloc[0][0])
print("Ending date: ", df.iloc[-1][0])
print("Duration: ", df.iloc[-1][0]-df.iloc[0][0])


# In[396]:


print("Min Date:",tesla_spec["Date"].min())
print("Max Date:",tesla_spec["Date"].max())


# In[397]:


tesla_spec.index=tesla_spec["Date"]


# In[399]:


tesla_spec



# In[400]:


tesla_spec.drop("Date",axis=1,inplace=True)


# In[402]:


tesla_spec



# In[403]:


results =tesla_spec.copy()


# # 1. What was the change in price of the stock overtime?
# 
# In this section we'll go over how to handle requesting stock information with pandas, and how to analyze basic attributes of a stock.

# # Data Analyzation & Visualization ✍🏻

# In[405]:


month_order= df.groupby(df['Date'].dt.strftime('%B'))[['Open','Close']].mean()
new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 
             'September', 'October', 'November', 'December']
month_order = month_order.reindex(new_order, axis=0)

month_order.plot(linewidth = '7.5' , figsize=(19, 10))
plt.ylabel('Open & Close')
plt.xlabel('Date')
plt.title("Sales Volume of Tesla")
plt.tight_layout()
plt.show()



# In[408]:


fig = go.Figure()

fig.add_trace(go.Bar(
    x=monthnew.index,
    y=monthnew['Open'],
    name='Stock Open Price',
    marker_color='grey'
))
fig.add_trace(go.Bar(
    x=monthnew.index,
    y=monthnew['Close'],
    name='Stock Close Price',
    marker_color='red'
))

fig.update_layout(barmode='group', xaxis_tickangle=-45, 
                  title='Stock Open & Close price on different months')
fig.show()


# In[162]:


plt.figure(figsize=(15, 6))
df['Open'].plot()
df['Close'].plot()
plt.ylabel(None)
plt.xlabel(None)
plt.title("Opening & Closing Price of Tesla")
plt.legend(['Open Price', 'Close Price'])
plt.tight_layout()
plt.show()


# In[163]:


df.groupby(df['Date'].dt.strftime('%B'))['Low'].min()
monthvise_high = df.groupby(df['Date'].dt.strftime('%B'))['High'].max()
monthvise_high = monthvise_high.reindex(new_order, axis=0)

monthvise_low = df.groupby(df['Date'].dt.strftime('%B'))['Low'].min()
monthvise_low = monthvise_low.reindex(new_order, axis=0)

fig = go.Figure()
fig.add_trace(go.Bar(
    x=monthvise_high.index,
    y=monthvise_high,
    name='Stock high Price',
    marker_color='rgb(0, 153, 204)'
))
fig.add_trace(go.Bar(
    x=monthvise_low.index,
    y=monthvise_low,
    name='Stock low Price',
    marker_color='rgb(255, 128, 0)'
))

fig.update_layout(barmode='group', 
                  title=' Monthwise High and Low stock price')
fig.show()


# In[164]:


plt.figure(figsize=(15, 6))
df['Volume'].plot()
plt.ylabel('Volume')
plt.xlabel(None)
plt.title("Sales Volume of Tesla")
plt.tight_layout()
plt.show()


# In[464]:


plt.figure(figsize=(15, 6))
df['Adj Close'].pct_change().hist(bins=50)
plt.ylabel('Daily Return')
plt.title(f'Tesla Daily Return')
plt.tight_layout()
plt.show()


# In[463]:


plt.figure(figsize=(20,7))
sns.lineplot(data=df,x="Date",y="Open",color="black",label="Open")
sns.lineplot(data=df,x="Date",y="Close",color="pink",label="Close")

plt.title("The relation between  Date of Open & Close value")


# In[167]:


plt.figure(figsize=(20,7))
sns.scatterplot(data=df,x="Date",y="Volume",)

plt.title("The relation between  Date of Volume")


# In[168]:


names = cycle(['Stock Open Price','Stock Close Price','Stock High Price','Stock Low Price'])

fig = px.line(df, x=df.Date, y=[df['Open'], df['Close'], 
                                          df['High'], df['Low']],
             labels={'Date': 'Date','Value':'Stock value'})
fig.update_layout(title_text='Stock analysis chart', font_size=15, font_color='black',legend_title_text='Stock Parameters')
fig.for_each_trace(lambda t:  t.update(name = next(names)))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)

fig.show()


# 
# ## Closing Price
# 
# The closing price for Tesla stock represents the final price at which Tesla shares are exchanged during the regular trading session. This figure serves as the standard reference point for investors, enabling them to monitor Tesla's performance across different timeframes.
# 

# In[169]:


closedf = df[['Date','Close']]
print("Shape of close dataframe:", closedf.shape)


# In[170]:


fig = px.line(closedf, x=closedf.Date, y=closedf.Close,labels={'Date':'Date','Close':'Close Stock'})
fig.update_traces(marker_line_width=2, opacity=0.8)
fig.update_layout(title_text='Stock close price chart', plot_bgcolor='white', font_size=15, font_color='black')
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()


# In[171]:


closedf = closedf[closedf['Date'] > '2020-08-16']
close_stock = closedf.copy()
print("Total data for prediction: ",closedf.shape[0])


# In[172]:


fig = px.line(closedf, x=closedf.Date, y=closedf.Close,labels={'Date':'Date','Close':'Close Stock'})
fig.update_traces(marker_line_width=2, opacity=0.8, marker_line_color='orange')
fig.update_layout(title_text='Considered period to predict Stock close price', plot_bgcolor='white', font_size=15, font_color='black')
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()


# In[173]:


plt.figure(figsize=(12,6))
plt.plot(tesla_data["Close"],color="blue");
plt.ylabel("Stock Price")
plt.title("Tesla Stock Price")
plt.xlabel("Time")
plt.show()


# In[174]:


ma_200_days = df.Close.rolling(200).mean()
plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days,'b')
plt.plot(df.Close,'g')
plt.show()


# # Analysis Results 🕵️
# Data consists of the monthly stock price of Tesla from 2010 to 2021. The data was structured in 7 main columns
# Date , Open  ,  Low ,  ,High,   Adj Close,  Volume ,  Close.
# 
# Tesla's stock witnessed remarkable growth and volatility, reflecting the company's dynamic performance and market sentiment. The open and close prices serve as crucial indicators for investors, offering insights into the stock's trading patterns, market sentiment, and overall performance.
# 
# We see the price Volume stable after 4-2021
# 
# 

# # Building a Machine Learning Model / Testing Model 🧬

# In[412]:


dataset = df["Close"]
dataset = pd.DataFrame(dataset)

data = dataset.values

data.shape


# In[413]:


scaler = MinMaxScaler(feature_range= (0, 1))
scaled_data = scaler.fit_transform(np.array(dataset).reshape(-1, 1))


# In[414]:


train_size = int(len(data)*.75)
test_size = len(data) - train_size

print("Train Size :",train_size,"Test Size :",test_size)

train_data = scaled_data[ :train_size , 0:1 ]
test_data = scaled_data[ train_size-60: , 0:1 ]


# In[415]:


train_data.shape, test_data.shape


# In[416]:


x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])


# In[417]:


x_train, y_train = np.array(x_train), np.array(y_train)


# In[418]:


x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# In[419]:


x_train.shape , y_train.shape


# In[420]:


model = Sequential([
    LSTM(50, return_sequences= True, input_shape= (x_train.shape[1], 1)),
    LSTM(64, return_sequences= False),
    Dense(32),
    Dense(16),
    Dense(1)
])

model.compile(optimizer= 'adam', loss= 'mse' , metrics= "mean_absolute_error")


# In[421]:


model.summary()



# In[422]:


callbacks = [EarlyStopping(monitor= 'loss', patience= 10 , restore_best_weights= True)]
history = model.fit(x_train, y_train, epochs= 100, batch_size= 32 , callbacks= callbacks )


# In[423]:


plt.plot(history.history["loss"])
plt.plot(history.history["mean_absolute_error"])
plt.legend(['Mean Squared Error','Mean Absolute Error'])
plt.title("Losses")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()


# In[426]:


x_test = []
y_test = []

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    y_test.append(test_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# In[427]:


x_test.shape , y_test.shape


# In[428]:


predictions = model.predict(x_test)

#inverse predictions scaling
predictions = scaler.inverse_transform(predictions)
predictions.shape


# In[429]:


y_test = scaler.inverse_transform([y_test])

RMSE = np.sqrt(np.mean( y_test - predictions )**2).round(2)
RMSE


# In[444]:


train = dataset.iloc[:train_size , 0:1]
test = dataset.iloc[train_size: , 0:1]
test['Predictions'] = predictions

plt.figure(figsize= (16, 6))
plt.title('Tesla Close Stock Price Prediction', fontsize= 18)
plt.xlabel('Date', fontsize= 18)
plt.ylabel('Close Price', fontsize= 18)
plt.plot(train['Close'], linewidth= 3)
plt.plot(test['Close'], linewidth= 3)
plt.plot(test["Predictions"], linewidth= 3)
plt.legend(['Train', 'Test', 'Predictions'])


# # Forecast Next 30 Days 📽️
# 

# In[445]:


from datetime import timedelta


# In[446]:


def insert_end(Xin, new_input):
    timestep = 60
    for i in range(timestep - 1):
        Xin[:, i, :] = Xin[:, i+1, :]
    Xin[:, timestep - 1, :] = new_input
    return Xin


# In[447]:


future = 30
forcast = []
Xin = x_test[-1 :, :, :]
time = []
for i in range(0, future):
    out = model.predict(Xin, batch_size=5)
    forcast.append(out[0, 0]) 
    print(forcast)
    Xin = insert_end(Xin, out[0, 0]) 
    time.append(pd.to_datetime(df.index[-1]) + timedelta(days=i))


# In[448]:


time


# In[449]:


forcasted_output = np.asanyarray(forcast)   
forcasted_output = forcasted_output.reshape(-1, 1) 
forcasted_output = scaler.inverse_transform(forcasted_output) 


# In[450]:


forcasted_output = pd.DataFrame(forcasted_output)
date = pd.DataFrame(time)
df_result = pd.concat([date,forcasted_output], axis=1)
df_result.columns = "Date", "Forecasted"


# In[451]:


df_result


# In[453]:


plt.figure(figsize=(16, 8))
plt.title('Tesla Close Stock Price Forecasting For Next 30 Days')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close' ,fontsize=18)
plt.plot(df['Close'])
plt.plot(df_result.set_index('Date')[['Forecasted']])


# In[ ]:





# In[459]:


x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

lst_output=[]
n_steps=time_step
i=0
pred_days = 30
while(i<pred_days):
    
    if(len(temp_input)>time_step):
        
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        
        yhat = model.predict(x_input, verbose=0)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
       
        lst_output.extend(yhat.tolist())
        i=i+1
        
    else:
        
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
               
print("Output of predicted next days: ", len(lst_output))


# In[460]:


last_days=np.arange(1,time_step+1)
day_pred=np.arange(time_step+1,time_step+pred_days+1)
print(last_days)
print(day_pred)


# In[461]:


temp_mat = np.empty((len(last_days)+pred_days+1,1))
temp_mat[:] = np.nan
temp_mat = temp_mat.reshape(1,-1).tolist()[0]

last_original_days_value = temp_mat
next_predicted_days_value = temp_mat

last_original_days_value[0:time_step+1] = scaler.inverse_transform(closedf[len(closedf)-time_step:]).reshape(1,-1).tolist()[0]
next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

new_pred_plot = pd.DataFrame({
    'last_original_days_value':last_original_days_value,
    'next_predicted_days_value':next_predicted_days_value
})

names = cycle(['Last 15 days close price','Predicted next 30 days close price'])

fig = px.line(new_pred_plot,x=new_pred_plot.index, y=[new_pred_plot['last_original_days_value'],
                                                      new_pred_plot['next_predicted_days_value']],
              labels={'value': 'Stock price','index': 'Timestamp'})
fig.update_layout(title_text='Compare last 15 days vs next 30 days',
                  plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')

fig.for_each_trace(lambda t:  t.update(name = next(names)))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()


# In[462]:


lstmdf=closedf.tolist()
lstmdf.extend((np.array(lst_output).reshape(-1,1)).tolist())
lstmdf=scaler.inverse_transform(lstmdf).reshape(1,-1).tolist()[0]

names = cycle(['Close price'])

fig = px.line(lstmdf,labels={'value': 'Stock price','index': 'Timestamp'})
fig.update_layout(title_text='Plotting whole closing stock price with prediction',
                  plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Stock')

fig.for_each_trace(lambda t:  t.update(name = next(names)))

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()

