import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

start = '2010-01-01'
end = '2022-01-01'

st.title('Stock Trend Prediction')
usr_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = data.DataReader(usr_input,'yahoo',start,end)

st.subheader('Data from 2010-2022')
st.write(df.describe())

st.subheader('Closing Price VS Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price VS Time Chart 100 Moving Average')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100)
st.pyplot(fig)

st.subheader('Closing Price VS Time Chart 200 Moving Average')
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100,'r')
plt.plot(ma200,'g')
st.pyplot(fig)

data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_test = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])

scalar = MinMaxScaler(feature_range=(0,1))

model = load_model('keras_model.h5')

past100 = data_train.tail(100)
final_df = past100.append(data_test, ignore_index = True)
input_data = scalar.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i,0])

x_test,y_test = np.array(x_test), np.array(y_test)

y_pred = model.predict(x_test)

scale = scalar.scale_

scale_factor = 1/scale[0]
y_pred = y_pred*scale_factor
y_test = y_test*scale_factor

st.subheader('Predictions VS Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label = 'Original Price')
plt.plot(y_pred,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig2)