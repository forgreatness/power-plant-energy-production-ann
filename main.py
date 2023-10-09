import numpy as np;
import pandas as pd;
import tensorflow as tf;

# In this project, we will be building an ANN for predicting the energy output of a power plant. Using data obtain from UCI Machine Learning Repository
# Data Link: https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant
# Data Consist of 4 features:
    # a. hourly average ambient temp
    # b. pressure
    # c. relative humidity
    # d. exhaust vacuum
# Output looking to predict is the: energy 

# Step #1 Data Preprocessing
    # Read data
    # Divide data into input and output
    # Split the both output data and input data into training data and testing data
data = pd.read_excel('./power_plant_production_records.xlsx')
input_data = data.iloc[:, :-1].values
output_data = data.iloc[:, -1].values

from sklearn.model_selection import train_test_split
training_input_data, testing_input_data, training_ouput_data, testing_output_data = train_test_split(input_data, output_data, test_size=0.2, random_state=0)

# Step #2 Building the ANN
    # Draw out what the ANN should look like
    # Determine how many neuron each hidden layer will have, and choose the best activation function for that layer. (Read about ann to know what activation function to use)
    # Add hidden layers
    # Add output layers 
ann = tf.keras.models.Sequential()
ann.add(layer=tf.keras.layers.Dense(units=6, activation="relu"))
ann.add(layer=tf.keras.layers.Dense(units=6, activation="relu"))
ann.add(tf.keras.layers.Dense(units = 1))

# Step #3 
    # Determine what cost function to use for the optimizer, optimizer is a function that is use to update weight
    # Traing the ann for a specific amount of epoch, and determine what the batch size is. 
adam_optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.mean_squared_error
ann.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
ann.fit(training_input_data, training_ouput_data, epochs=50, batch_size=50)

    
# Step #4 Predicting data for each case
predicted_output = ann.predict(testing_input_data)
np.set_printoptions(precision=2) #setting output value of pandas to 2 decimal (pandas is build on top of numpy "remember")
print(np.concatenate((predicted_output.reshape(len(predicted_output),1), testing_output_data.reshape(len(testing_output_data), 1)), 1))

# We are building an ANN for regression so we will be using a certain loss function and optimizer
# adamOptimizer = tf.keras.optimizers.Adam()

