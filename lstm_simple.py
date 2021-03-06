import pandas as pd
import datetime
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
sns.set()
#%matplotlib inline
import plotly.offline as py
#py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import statsmodels.api as sm
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from IPython.display import display

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import warnings
# import the_module_that_warns

warnings.filterwarnings("ignore")

from fbprophet import Prophet


## for Deep-learing:
import keras
from keras.layers import Dense, Activation,Reshape
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD,Adadelta,Adam,RMSprop 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import itertools
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import model_from_json


def load_data(datapath):
	data_type = {'store': 'int8', 'item': 'int8', 'sales': 'int16'}
	data = pd.read_csv(datapath, index_col=['date', 'store','item'])
	# Dimensions
	print('Shape:', data.shape)
	# Set of features we have are: date, store, and item
	display(data.sample(10))
	return data

# one LSTM + Dense , stateless
def fit_lstm(train_input, train_output, batch_size, nb_epoch, neurons):
	model = Sequential()
#	model.add(LSTM(neurons, activation='hard_sigmoid', batch_input_shape=(batch_size, train_input.shape[1], train_input.shape[2]), return_sequences=True, stateful=True,recurrent_dropout=0.2))
	model.add(LSTM(neurons, activation='hard_sigmoid', batch_input_shape=(batch_size, train_input.shape[1], train_input.shape[2]), recurrent_dropout=0.2))
	model.add (Dense(500) )
	model.add(Reshape((10,50)) )
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(train_input, train_output, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
		model.reset_states()
	return model

def refit_lstm(train_input, train_output, batch_size, nb_epoch, model):
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(train_input, train_output, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
		model.reset_states()
	return model

train_df = load_data('./input/train.csv')

# Prepare train data

now = datetime.now()

print('Train Dataset[start]:',now.hour,':',now.minute,':',now.second)

dates = train_df.index.levels[0].unique()
sample_size = dates.size-1

stores = train_df.index.levels[1].unique()
items = train_df.index.levels[2].unique()

# shift dataset to calculate difference (avoid trends in data)
diff_train_df = (train_df -np.roll(train_df,1,axis=0)).astype(float)
# fill 1st vector with seros


diff_train_df.loc['2013-01-01'] = 0

#normalize data sets

#train_max= np.max(abs(diff_train_df))
#test_max= np.max(abs(diff_test_df))


#diff_train_df = diff_train_df/train_max


#scaler = MinMaxScaler(feature_range=(-1, 1))
#scaler = scaler.fit(diff_train_df)
#scalled_train_df = scaler.transform(diff_train_df)
input_set = np.zeros((sample_size,10,50))
output_set = np.zeros((sample_size,10,50))

# initial iput set shape 10x50

input = np.hstack([diff_train_df.loc[dates[0],1],
	diff_train_df.loc[dates[0],2],
	diff_train_df.loc[dates[0],3],
	diff_train_df.loc[dates[0],4],
	diff_train_df.loc[dates[0],5],
	diff_train_df.loc[dates[0],6],
	diff_train_df.loc[dates[0],7],
	diff_train_df.loc[dates[0],8],
	diff_train_df.loc[dates[0],9],
	diff_train_df.loc[dates[0],10]
	])
iter = 0
for date in dates[1:1826]: 
	# print(date)
	input_set[iter] = input.transpose()
	output = np.hstack([diff_train_df.loc[date,1],
		diff_train_df.loc[date,2],
		diff_train_df.loc[date,3],
		diff_train_df.loc[date,4],
		diff_train_df.loc[date,5],
		diff_train_df.loc[date,6],
		diff_train_df.loc[date,7],
		diff_train_df.loc[date,8],
		diff_train_df.loc[date,9],
		diff_train_df.loc[date,10]
		])
	output_set[iter] = output.transpose()
	iter += 1
	input = output

#normalize data sets

train_max= np.max(abs(diff_train_df))
diff_train_df = diff_train_df/train_max

now = datetime.now()

print('Train Dataset[end]:',now.hour,':',now.minute,':',now.second)

#now I have inputs in input_set
# and outputs (shifted by 1) in output_set

#Let's split it into train and 
X_train, X_test, y_train, y_test = train_test_split(input_set, output_set, test_size = 0.3, random_state = 0)


# scale sets

#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

#Sequence classification with LSTM

#lstm_model = fit_lstm(X_train,y_train, 1, 50, 1000)
#lstm_model = fit_lstm(X_train[0:1270],y_train[0:1270], 127, 50, 500)

scores = lstm_model.evaluate(X_train,y_train, verbose=0,batch_size=1)
#print("%s: %.2f%%" % (lstm_model.metrics_names[1], scores[1]*100))
# serialize model to JSON
model_json = lstm_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
lstm_model.save_weights("model.h5")
print("Saved model to disk")


# load json and create model
json_file = open('model_lstm.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_lstm.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

scores = loaded_model.evaluate(X_train,y_train, verbose=0,batch_size=1)
#lstm_model = refit_lstm(X_train[0:1270],y_train[0:1270], 127, 50, lstm_model)
score = loaded_model.evaluate(X_train, y_train, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

