
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from train import Model, mse, gd, sgd, momentum # import our math


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

# Add nonlinear tire degradation features
data["tyre_age_squared"] = data["tyre_age"] ** 2

data["soft_tyre_age"] = (
        (data["compound"] == "SOFT").astype(float)
        * data["tyre_age"]
)


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
gd_losses, gd_times = gd(gd_model, X_train, y_train, learning_rate=0.05, epochs=200)#gd learning rate

"""
compare training mse before and after optimization
#get root mse of the test data we set apart with our regression after testing
"""
gd_initial_rmse = np.sqrt(gd_initial_loss)
gd_final_rmse = np.sqrt(mse(y_train, gd_model.predict(X_train)))
gd_test_rmse = np.sqrt(mse(y_test, gd_model.predict(X_test)))

"""
make seperate SGD model
seperate model w/ same initial weights and biases as gd
trained on same laps, updated after each indv. lap
"""
sgd_model = Model(n_features=X_train.shape[1])
sgd_model.bias = np.mean(y_train)

sgd_initial_rmse = np.sqrt(mse(y_train, sgd_model.predict(X_train)))
sgd_losses, sgd_times = sgd(sgd_model, X_train, y_train, learning_rate=0.0002, epochs=200) #sgd learning rate
sgd_final_rmse = np.sqrt(mse(y_train, sgd_model.predict(X_train)))
sgd_test_rmse = np.sqrt(mse(y_test, sgd_model.predict(X_test)))

"""
seperate Momentum model


s
"""
momentum_model = Model(n_features=X_train.shape[1])
momentum_model.bias = np.mean(y_train)

momentum_initial_rmse = np.sqrt(mse(y_train, momentum_model.predict(X_train)))
momentum_losses = momentum(momentum_model, X_train, y_train, learning_rate=0.01, epochs=200)


momentum_final_rmse = np.sqrt(mse(y_train, momentum_model.predict(X_train)))
momentum_test_rmse = np.sqrt(mse(y_test, momentum_model.predict(X_test)))



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
print()
print("MOMENTUM:")
print(f"Initial training RMSE: {momentum_initial_rmse} seconds")
print(f"Final training RMSE: {momentum_final_rmse} seconds")
print(f"Test RMSE: {momentum_test_rmse} seconds")



gd_rmse = np.sqrt(gd_losses)
sgd_rmse = np.sqrt(sgd_losses)
epochs = range(1, len(gd_rmse) + 1)
plt.figure(figsize=(7, 4))
plt.plot(epochs, gd_rmse, label="GD")
plt.plot(epochs, sgd_rmse, label="SGD")
plt.xlabel("Epoch")
plt.ylabel("Training RMSE (seconds)")
plt.title("Convergence of GD and SGD")
plt.legend()
plt.tight_layout()
plt.savefig("convergence_plot.png", dpi=300)
plt.show()

#real time graph
plt.figure(figsize=(7, 4))
plt.plot(gd_times, gd_rmse, label="GD")
plt.plot(sgd_times, sgd_rmse, label="SGD")

plt.xlabel("Training Time (seconds)")
plt.ylabel("Training RMSE (seconds)")
plt.title("GD vs SGD: Convergence Over Time")

plt.legend()
plt.tight_layout()
plt.savefig("convergence_time.png", dpi=300)
plt.show()

print()
print("GD training time:", gd_times[-1], "seconds")
print("SGD training time:", sgd_times[-1], "seconds")



# GD LEARNING RATE EXPERIMENT
# Train fresh models with different learning rates.
# Keep everything else identical to compare convergence.

gd_rates = [0.001, 0.01, 0.03, 0.05, 0.1]

plt.figure(figsize=(7, 4))

for lr in gd_rates:
    test_model = Model(n_features=X_train.shape[1])
    test_model.bias = np.mean(y_train)
    initial_rmse = np.sqrt(mse(y_train, test_model.predict(X_train)))#Record the error before training
    losses, times = gd(test_model, X_train, y_train, learning_rate=lr, epochs=200)
    rmse = np.concatenate(([initial_rmse], np.sqrt(losses)))
    plt.plot(range(len(rmse)), rmse, label=f"LR = {lr}")

plt.xlabel("Epoch")
plt.ylabel("Training RMSE (seconds)")
plt.title("GD Convergence at Different Learning Rates")
plt.legend()
plt.tight_layout()
plt.savefig("gd_learning_rates.png", dpi=300)
plt.show()


# SGD learning rate plotting
sgd_rates = [0.01, 0.001, 0.0002, 0.0001]
plt.figure(figsize=(7, 4))

for lr in sgd_rates:
    test_model = Model(n_features=X_train.shape[1])
    test_model.bias = np.mean(y_train)
    initial_rmse = np.sqrt(mse(y_train, test_model.predict(X_train)))
    losses, times = sgd(test_model, X_train, y_train, learning_rate=lr, epochs=200)
    rmse = np.concatenate(([initial_rmse], np.sqrt(losses)))
    plt.plot(range(len(rmse)), rmse, label=f"LR ={lr}")

plt.xlabel("Epoch")
plt.ylabel("Training RMSE (seconds)")
plt.title("SGD Convergence at Different Learning Rates")
plt.legend()
plt.tight_layout()
plt.savefig("sgd_learning_rates.png", dpi=300)
plt.show()



# Find when each optimizer first reaches 0.65 seconds training RMSE.
for name, losses, times in [
    ("GD", gd_losses, gd_times),
    ("SGD", sgd_losses, sgd_times)
]:
    reached = np.where(np.sqrt(losses) <= 0.65)[0]

    if len(reached) > 0:
        i = reached[0]
        print(f"{name}: {i + 1} epochs, {times[i]:.4f} seconds")
    else:
        print(f"{name}: Did not reach 0.65 RMSE")
