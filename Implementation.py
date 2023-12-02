import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset

dataset = pd.read_csv("Credit_card_Applications.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

print("Shape: ", dataset.shape)

# Feature Scaling

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(X)

# Implementing SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len=15, learning_rate=0.5, sigma=1.0) # Sigma is the radius
som.random_weights_init(X)
som.train_random(X, num_iteration=100)

# Visualizing the Results
# We need to get the MID (Mean Interneuron Distance)

from pylab import bone, pcolor, plot, show, colorbar
bone() # to initialize the window
pcolor(som.distance_map().T)
colorbar()
markers = ['o','s']
colors = ['r', 'g']

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5,
         markers[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)

show()

# Catching the Frauds

mappings = som.win_map(X)
frauds = np.concatenate((mappings[(1,8)], mappings[(5,2)], mappings[(8,6)]), axis = 0)

frauds = scaler.inverse_transform(frauds)


### Moving from Unsupervised to Supervised

# Creating the matrix of features

customers = dataset.iloc[:,1:].values

# Creating the target variable

is_fraud = np.zeros(len(dataset))


for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1
        


### Building the ANN Model

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)


from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units = 6, kernel_initializer='uniform', activation = 'relu', input_dim = 15))
model.add(Dense(units = 6, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(units = 1, kernel_initializer='uniform', activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics = ['accuracy'])

model.fit(customers, is_fraud, batch_size=5, epochs=10)

### Predicting the Probabilities of frauds

predicted = model.predict(customers)

predicted_sorted = np.concatenate((dataset.iloc[:,0:1].values,predicted),axis = 1)
predicted_sorted = predicted_sorted[predicted_sorted[:,1].argsort()]


