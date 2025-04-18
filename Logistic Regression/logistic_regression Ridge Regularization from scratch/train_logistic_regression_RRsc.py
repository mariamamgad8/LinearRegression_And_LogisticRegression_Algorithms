import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#-----------RR from scratch-----------

# Load training data
train_df = pd.read_csv('training_shooter_game_cheater_dataset_500.csv')
X_train = train_df.drop(columns=['Cheater']).values
y_train = train_df['Cheater'].values.reshape(-1, 1)

# Normalize features   ============>AKA "scalling"
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))  # Add bias term

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Ridge-regularized cost
def compute_cost_ridge(X, y, weights, lambda_):
    m = len(y)
    h = sigmoid(np.dot(X, weights))
    epsilon = 1e-5  #======> 0.00001 to Avoid infinity =====> log(0)
    cost = -(1/m) * (np.dot(y.T, np.log(h + epsilon)) + np.dot((1 - y).T, np.log(1 - h + epsilon)))
    reg_term = (lambda_ / (2 * m)) * np.sum(np.square(weights[1:]))  # exclude bias from regularization
    return cost[0, 0] + reg_term

# Ridge-regularized logistic regression
def logistic_regression_ridge(X, y, learning_rate=0.1, epochs=1000, lambda_=1.0):
    m, n = X.shape
    weights = np.zeros((n, 1))
    costs = []
    for i in range(epochs):
        h = sigmoid(np.dot(X, weights))
        gradient = (1/m) * np.dot(X.T, (h - y))
        gradient[1:] += (lambda_ / m) * weights[1:]  # regularize all but the bias term
        weights -= learning_rate * gradient
        if i % 10 == 0:
            costs.append(compute_cost_ridge(X, y, weights, lambda_))
    plt.plot(range(0, epochs, 10), costs)
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.title("Ridge-Regularized Cost vs. Epochs")
    plt.show()
    return weights

# Train the model
weights = logistic_regression_ridge(X_train, y_train, lambda_=1.0)

# Predict and evaluate
def predict(X, weights):
    return (sigmoid(np.dot(X, weights)) >= 0.5).astype(int)

y_pred_train = predict(X_train, weights)
train_accuracy = (y_pred_train == y_train).mean()
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

# Save weights and normalization parameters
np.save('weights.npy', weights)
np.save('mean.npy', mean)
np.save('std.npy', std)

print("Training complete. Saved: weights.npy, mean.npy, std.npy.")
