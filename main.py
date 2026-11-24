
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from train import Model, mse, gd, sgd, momentum, adam# import our math


"""
load f1 data
 
Data:
openf1 API
Race: 2024 Bahrain GP

we have 1065 recorded laps, each row is a driver
Input features:
    - driver_number = car/drive identifier, we have 
    - lap_number = lap identifier
    - compound = tire compound, 2 options, soft or hard
    - tyre_age = age of tires in laps; I think euro spells tire as tyre
    
Target var:
    lap_time_seconds = laptime in seconds, this is our control/target
"""
data_path = Path(__file__).resolve().parent/"bahrain_2024_clean.csv"
data = pd.read_csv(data_path)#panda dataframe/table

# Add nonlinear tire degradation features
#data["tyre_age_squared"] = data["tyre_age"] ** 2

#data["soft_tyre_age"] = (
  #      (data["compound"] == "SOFT").astype(float)
   #     * data["tyre_age"]
#)


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

split = int(len(X) * 0.8)#take 80% for training and validation, int cause we dont want fractional data
train_split = int(split * 0.8)#take 80% of training and validation for training, rest for validation

train_indices = indices[:train_split]
val_indices = indices [train_split:split]
test_indices = indices[split:]

X_train = X[train_indices]
X_test = X[test_indices]
X_val = X[val_indices]
y_train = y_realized[train_indices]
y_test = y_realized[test_indices]
y_val = y_realized[val_indices]


#standardize input features for gd and sgd to make input ranges closer
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

std[std == 0] = 1.0 #avoid dividing by 0
X_train = (X_train-mean) / std
X_test = (X_test-mean)/ std
X_val = (X_val-mean) / std


"""
helper function to check when validation RMSE Stabilizes

calc percent diff. between sequential epochs,
find first 10 epoch window with avg change of below .1%,
check rmse stayed within 1% of final value at end, ie, 
make sure optimization method actually converged and stayed
"""
def convergence_epoch(val_losses, window=10, threshold=0.1):
    rmse = np.sqrt(np.array(val_losses))
    changes = 100 * np.abs(np.diff(rmse)) / rmse[:-1]# check epochs rate of change delta,

    for start in range(len(changes) - window + 1):
        avg_change = np.mean(changes[start:start + window])#take avg
        end = start + window# Last epoch included in this window
        near_final = np.all(np.abs(rmse[end:] - rmse[-1]) / rmse[-1] < 0.01)
        if avg_change < threshold and near_final:
            return end + 1
    return None


# GD learning rate testing, decision, and plotting
gd_rates = np.arange(0.005, 0.151, 0.005)
results = []

plt.figure(figsize=(7, 4))

for lr in gd_rates:
    #train fresh model for-each learning rate w/ identical setup
    test_model = Model(n_features=X_train.shape[1])
    test_model.bias = np.mean(y_train)
    losses, times, val_losses = gd(test_model, X_train, y_train, learning_rate=lr, epochs=200, X_val=X_val, y_val=y_val)
    val_rmse = np.sqrt(val_losses)
    convergence = convergence_epoch(val_losses)
    results.append((lr, val_rmse[-1], convergence))#store learning rate, final rmse, and convergence epoch
    plt.plot(range(1, len(val_rmse) + 1), val_rmse, label=f"LR = {lr:.4f}")
plt.xlabel("Epoch")
plt.ylabel("Validation RMSE (seconds)")
plt.title("GD Learning Rate Comparison")
plt.legend(fontsize=7, ncol=3)
plt.tight_layout()
plt.savefig("gd_learning_rates.png", dpi=300)
plt.show()

best_rmse = min(result[1] for result in results)# find lowest rmse from all tested learning rates
valid = [
    result for result in results
    if result[1] <= best_rmse * 1.01 and result[2] is not None # check if within 1% of best rmse and if we met convergence criteria
]
if valid:
    best = min(valid, key=lambda result: result[2])# find result w/ smallest convergence epoch
    gd_rate = best[0]
    print(f"Best GD learning rate: {gd_rate}")
    print(f"Validation RMSE: {best[1]}")
    print(f"Convergence epoch: {best[2]}")
    print()
else:
    raise ValueError("No GD learning rate met the convergence criteria.")



# SGD learning rate testing and plotting
sgd_rates = np.arange(0.0001, 0.0011, 0.0001)
results = []

plt.figure(figsize=(7, 4))

for lr in sgd_rates:
    test_model = Model(n_features=X_train.shape[1])
    test_model.bias = np.mean(y_train)
    losses, times, val_losses = sgd(test_model, X_train, y_train, learning_rate=lr, epochs=200, X_val=X_val, y_val=y_val)
    val_rmse = np.sqrt(val_losses)
    convergence = convergence_epoch(val_losses)
    results.append((lr, val_rmse[-1], convergence))
    plt.plot(range(1, len(val_rmse) + 1), val_rmse, label=f"LR = {lr:.4f}")

plt.xlabel("Epoch")
plt.ylabel("Validation RMSE (seconds)")
plt.title("SGD Learning Rate Comparison")
plt.legend(fontsize=7, ncol=3)
plt.tight_layout()
plt.savefig("sgd_learning_rates.png", dpi=300)
plt.show()

best_rmse = min(result[1] for result in results)#find lowest rmse from all tested learning rates
valid = [
    result for result in results
    if result[1] <= best_rmse * 1.01 and result[2] is not None #check if within 1% of best rmse and met convergence criteria
]

if valid:
    best = min(valid, key=lambda result: result[2])#find result w/ smallest convergence epoch
    sgd_rate = best[0]
    print(f"Best SGD learning rate: {sgd_rate}")
    print(f"Validation RMSE: {best[1]}")
    print(f"Convergence epoch: {best[2]}")
    print()
else:
    raise ValueError("No SGD learning rate met the convergence criteria.")



# Momentum learning rate testing and plotting
momentum_rates = np.arange(0.001, 0.051, 0.001)
results = []

plt.figure(figsize=(8, 5))

for lr in momentum_rates:
    #train fresh model for each learning rate w/ identical setup
    test_model = Model(n_features=X_train.shape[1])
    test_model.bias = np.mean(y_train)
    losses, times, val_losses = momentum(test_model, X_train, y_train, learning_rate=lr, epochs=200, X_val=X_val, y_val=y_val)
    val_rmse = np.sqrt(val_losses)
    convergence = convergence_epoch(val_losses)
    results.append((lr, val_rmse[-1], convergence))#store learning rate, final rmse, and convergence epoch
    plt.plot(range(1, len(val_rmse) + 1), val_rmse, label=f"LR = {lr:.3f}")

plt.xlabel("Epoch")
plt.ylabel("Validation RMSE (seconds)")
plt.title("Momentum Learning Rate Comparison")
plt.legend(fontsize=7, ncol=3)
plt.tight_layout()
plt.savefig("momentum_learning_rates.png", dpi=300)
plt.show()

best_rmse = min(result[1] for result in results)#find lowest rmse from all tested learning rates
valid = [
    result for result in results
    if result[1] <= best_rmse * 1.01 and result[2] is not None #check if within 1% of best rmse and met convergence criteria
]

if valid:
    best = min(valid, key=lambda result: result[2])#find result w/ smallest convergence epoch
    momentum_rate = best[0]
    print(f"Best Momentum learning rate: {momentum_rate}")
    print(f"Validation RMSE: {best[1]}")
    print(f"Convergence epoch: {best[2]}")
    print()
else:
    raise ValueError("No Momentum learning rate met the convergence criteria.")

# Adam learning rate testing and plotting
adam_rates = np.arange(0.001, 0.051, 0.001)
results = []

plt.figure(figsize=(8, 5))

for lr in adam_rates:
    #train fresh model for each learning rate w/ identical setup
    test_model = Model(n_features=X_train.shape[1])
    test_model.bias = np.mean(y_train)
    losses, times, val_losses = adam(test_model, X_train, y_train, learning_rate=lr, epochs=200, X_val=X_val, y_val=y_val)
    val_rmse = np.sqrt(val_losses)
    convergence = convergence_epoch(val_losses)
    results.append((lr, val_rmse[-1], convergence))#store learning rate, final rmse, and convergence epoch
    plt.plot(range(1, len(val_rmse) + 1), val_rmse, label=f"LR = {lr:.3f}")

plt.xlabel("Epoch")
plt.ylabel("Validation RMSE (seconds)")
plt.title("Adam Learning Rate Comparison")
plt.legend(fontsize=7, ncol=3)
plt.tight_layout()
plt.savefig("adam_learning_rates.png", dpi=300)
plt.show()

best_rmse = min(result[1] for result in results)#find lowest rmse from all tested learning rates
valid = [
    result for result in results
    if result[1] <= best_rmse * 1.01 and result[2] is not None #check if within 1% of best rmse and met convergence criteria
]

if valid:
    best = min(valid, key=lambda result: result[2])#find result w/ smallest convergence epoch
    adam_rate = best[0]
    print(f"Best Adam learning rate: {adam_rate}")
    print(f"Validation RMSE: {best[1]}")
    print(f"Convergence epoch: {best[2]}")
    print()
else:
    raise ValueError("No Adam learning rate met the convergence criteria.")

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
gd_losses, gd_times, gd_val_losses = gd(gd_model, X_train, y_train, learning_rate=gd_rate, epochs=200, X_val=X_val, y_val=y_val)#gd learning rate

#compare training mse before and after optimization
#get root mse of the test data we set apart with our regression after testing
gd_initial_rmse = np.sqrt(gd_initial_loss)
gd_final_rmse = np.sqrt(mse(y_train, gd_model.predict(X_train)))
gd_test_rmse = np.sqrt(mse(y_test, gd_model.predict(X_test)))
gd_val_rmse = np.sqrt(mse(y_val, gd_model.predict(X_val)))

"""
make seperate SGD model
seperate model w/ same initial weights and biases as gd
trained on same laps, updated after each indv. lap
"""
sgd_model = Model(n_features=X_train.shape[1])
sgd_model.bias = np.mean(y_train)

sgd_initial_rmse = np.sqrt(mse(y_train, sgd_model.predict(X_train)))
sgd_losses, sgd_times, sgd_val_losses = sgd(sgd_model, X_train, y_train, learning_rate=sgd_rate, epochs=200, X_val=X_val, y_val=y_val) #sgd learning rate

sgd_final_rmse = np.sqrt(mse(y_train, sgd_model.predict(X_train)))
sgd_test_rmse = np.sqrt(mse(y_test, sgd_model.predict(X_test)))
sgd_val_rmse = np.sqrt(mse(y_val, sgd_model.predict(X_val)))

"""
make seperate Momentum model
"""
momentum_model = Model(n_features=X_train.shape[1])
momentum_model.bias = np.mean(y_train)

momentum_initial_rmse = np.sqrt(mse(y_train, momentum_model.predict(X_train)))
momentum_losses, momentum_times, momentum_val_losses = momentum(momentum_model, X_train, y_train, learning_rate=momentum_rate, epochs=200, X_val=X_val, y_val=y_val)

momentum_final_rmse = np.sqrt(mse(y_train, momentum_model.predict(X_train)))
momentum_test_rmse = np.sqrt(mse(y_test, momentum_model.predict(X_test)))
momentum_val_rmse = np.sqrt(mse(y_val, momentum_model.predict(X_val)))



"""
make seperate adam model
"""
adam_model = Model(n_features=X_train.shape[1])
adam_model.bias = np.mean(y_train)

adam_initial_rmse = np.sqrt(mse (y_train, adam_model.predict(X_train)))
adam_losses, adam_times, adam_val_losses = adam(adam_model, X_train, y_train, learning_rate=adam_rate, epochs=200, X_val=X_val, y_val=y_val)

adam_final_rmse = np.sqrt(mse(y_train, adam_model.predict(X_train)))
adam_test_rmse = np.sqrt(mse(y_test, adam_model.predict(X_test)))
adam_val_rmse = np.sqrt(mse(y_val, adam_model.predict(X_val)))



print("Training laps:", len(y_train))
print("Validation laps:", len(y_val))
print("Testing laps:", len(y_test))
print("Input features:", X_train.shape[1])
print()
print("GRADIENT DESCENT:")
print(f"Initial training rmse: {gd_initial_rmse} seconds")
print(f"Final training rmse: {gd_final_rmse} seconds")
print(f"Validation rmse: {gd_val_rmse} seconds")
print(f"Test rmse: {gd_test_rmse} seconds")
print()
print("STOCHASTIC GRADIENT DESCENT:")
print(f"Initial training rmse: {sgd_initial_rmse} seconds")
print(f"Final training rmse: {sgd_final_rmse} seconds")
print(f"Validation rmse: {sgd_val_rmse} seconds")
print(f"Test rmse: {sgd_test_rmse} seconds")
print()
print("MOMENTUM:")
print(f"Initial training rmse: {momentum_initial_rmse} seconds")
print(f"Final training rmse: {momentum_final_rmse} seconds")
print(f"Validation rmse: {momentum_val_rmse} seconds")
print(f"Test rmse: {momentum_test_rmse} seconds")
print()
print("ADAM:")
print(f"Initial training rmse: {adam_initial_rmse} seconds")
print(f"Final training rmse: {adam_final_rmse} seconds")
print(f"Validation rmse: {adam_val_rmse} seconds")
print(f"Test rmse: {adam_test_rmse} seconds")

print()
print("Gradient Descent training time:", gd_times[-1], "seconds")
print("Stochastic Gradient Descent training time:", sgd_times[-1], "seconds")
print("Momentum training time:", momentum_times[-1], "seconds")
print("Adam training time:", adam_times[-1], "seconds")



























# Find when each optimizer first reaches 0.65 seconds training RMSE.
for name, losses, times in [
    ("GD", gd_losses, gd_times),
    ("SGD", sgd_losses, sgd_times)
]:
    reached = np.where(np.sqrt(losses) <= 0.675)[0]

    if len(reached) > 0:
        i = reached[0]
        print(f"{name}: {i + 1} epochs, {times[i]:.4f} seconds")
    else:
        print(f"{name}: Did not reach 0.65 RMSE")


# Compare validation convergence across all four optimizers
plt.figure(figsize=(8, 5))

epochs = range(1, len(gd_val_losses) + 1)

plt.plot(epochs, np.sqrt(gd_val_losses), label="GD")
plt.plot(epochs, np.sqrt(sgd_val_losses), label="SGD")
plt.plot(epochs, np.sqrt(momentum_val_losses), label="Momentum")
plt.plot(epochs, np.sqrt(adam_val_losses), label="Adam")

plt.xlabel("Epoch")
plt.ylabel("Validation RMSE (seconds)")
plt.title("Optimizer Comparison: Validation Convergence")

plt.legend()
plt.tight_layout()
plt.savefig("optimizer_comparison.png", dpi=300)
plt.show()













