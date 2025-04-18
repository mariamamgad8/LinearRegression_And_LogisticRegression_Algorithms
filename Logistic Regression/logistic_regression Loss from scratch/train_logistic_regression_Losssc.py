import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
#-----------Loss from scratch-----------

# Load training data
train_df = pd.read_csv('training_shooter_game_cheater_dataset_500.csv')

# Separate features and label
X_train = train_df.drop(columns=['Cheater']).values
y_train = train_df['Cheater'].values.reshape(-1, 1)

# Normalize features and save mean/std
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std

# Add bias column
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function to minimize
def compute_cost(weights, X, y):
    weights = weights.reshape(-1, 1)
    m = len(y)
    h = sigmoid(X @ weights)
    epsilon = 1e-5
    cost = -(1/m) * (y.T @ np.log(h + epsilon) + (1 - y).T @ np.log(1 - h + epsilon))
    return cost.flatten()[0]

# Optimization
# Initial guess for weights (must be 1D for scipy)
initial_weights = np.zeros(X_train.shape[1])

# Minimize cost
res = minimize(fun=compute_cost,
               x0=initial_weights,
               args=(X_train, y_train),
               method='BFGS',
               options={'maxiter': 1000, 'disp': True})

# Final weights (reshape to 2D for later use)
weights = res.x.reshape(-1, 1)


# Prediction function
def predict(X, weights):
    return (sigmoid(np.dot(X, weights)) >= 0.5).astype(int)

y_pred_train = predict(X_train, weights)
train_accuracy = (y_pred_train == y_train).mean()
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

# Save model
np.save('weights.npy', weights)
np.save('mean.npy', mean)
np.save('std.npy', std)

print("Training complete. Saved: weights.npy, mean.npy, std.npy.")
