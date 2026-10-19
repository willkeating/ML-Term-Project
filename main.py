
from pathlib import Path
import numpy as np
import pandas as pd

from train import Model, mse, gd, sgd# import our math


"""
load f1 data
 
Data:
openf1 API
Race: 2024 Bahrain GP

we have 1065 recorded laps, each row is a driver
Input features:
    driver_number = car/drive identifier, we have 
    lap_number = lap identifier
    compound = tire compound, 2 options, soft or hard
    tyre_age = age of tires in laps; I think euro spells tire as tyre
    
Target var:
    lap_time_seconds = laptime in seconds, this is our control/target
    
    
"""
data_path = Path(__file__).resolve().parent/"bahrain_2024_clean.csv"
data = pd.read_csv(data_path)#panda dataframe/table


#seperate our y_realized ie target from the input features
X = data.drop(columns=["lap_time_seconds"])
y_realized = data["lap_time_seconds"].to_numpy(dtype=float)


# replace driver # and tire compound cols w/ individual cols for each driver and each compound. 24 vars/cols total
# This allows the model to learn separate effects for each category.
X = pd.get_dummies(X, columns=["driver_number", "compound"], dtype=float)
X = X.to_numpy(dtype=float)#conv. table to numPy matrix


# Randomly shuffle laps, use seed so we can reproduce.
rng = np.random.default_rng(42)
indices = rng.permutation(len(X))
split = int(len(X) * 0.8)#Use 80% to train model use 20% to test predictions, can tune if wanted
train_indices = indices[:split]
test_indices = indices[split:]
X_train = X[train_indices]
X_test = X[test_indices]
y_train = y_realized[train_indices]
y_test = y_realized[test_indices]



#standardize input features for gd and sgd to make input ranges closer
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

std[std == 0] = 1.0 #avoid dividing by 0
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std



"""
setup linear regression
 - make one weight per encoded feature
 - start bias at avg training lap time
"""
gd_model = Model(n_features=X_train.shape[1])
gd_model.bias = np.mean(y_train)


"""
Train w/ GD
 - record initial mse and run 200 epochs of batch gd
 - each iter adjusts weights and bias to minimize error
"""
gd_initial_loss = mse(y_train, gd_model.predict(X_train))
gd_losses = gd(gd_model, X_train, y_train, learning_rate=0.05, epochs=200)

"""
compare training mse before and after optimization
#get root mse of the test data we set apart with our regression after testing
"""
gd_initial_rmse = np.sqrt(gd_initial_loss)
gd_final_rmse = np.sqrt(mse(y_train, gd_model.predict(X_train)))
gd_test_rmse = np.sqrt(mse(y_test, gd_model.predict(X_test)))

"""
make seperate SGD model
"""
sgd_model = Model(n_features=X_train.shape[1])
sgd_model.bias = np.mean(y_train)

sgd_initial_rmse = np.sqrt(mse(y_train, sgd_model.predict(X_train)))
sgd_losses = sgd(sgd_model, X_train, y_train, learning_rate=0.01, epochs=200)
sgd_final_rmse = np.sqrt(mse(y_train, sgd_model.predict(X_train)))
sgd_test_rmse = np.sqrt(mse(y_test, sgd_model.predict(X_test)))



"""
Train w/ SGD
seperate model w/ same initial weights and biases as gd
trained on same laps, updated after each indv. lap
"""




print("Training laps:", len(y_train))
print("Testing laps:", len(y_test))
print("Input features:", X_train.shape[1])
print()
print("GRADIENT DESCENT:")
print(f"Initial training root mean squared error: {gd_initial_rmse} seconds")
print(f"Final training root mean squared error: {gd_final_rmse} seconds")
print(f"Test root mean squared error: {gd_test_rmse} seconds")
print()
print("STOCHASTIC GRADIENT DESCENT:")
print(f"Initial training root mean squared error: {sgd_initial_rmse} seconds")
print(f"Final training root mean squared error: {sgd_final_rmse} seconds")
print(f"Test root mean squared error: {sgd_test_rmse} seconds")

