import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#-----------BGD from scratch-----------


# Load training data
train_df = pd.read_csv('training_shooter_game_cheater_dataset_500.csv')

# Separate features and label
X_train = train_df.drop(columns=['Cheater']).values
y_train = train_df['Cheater'].values.reshape(-1, 1)

# Normalize features and save mean/std ============>AKA "scalling"
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std

# Add bias column
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function
def compute_cost(X, y, weights):
    m = len(y)
    h = sigmoid(np.dot(X, weights))
    epsilon = 1e-5  #======> 0.00001 to Avoid infinity =====> log(0)
    cost = -(1/m) * (np.dot(y.T, np.log(h + epsilon)) + np.dot((1 - y).T, np.log(1 - h + epsilon)))
    return cost[0, 0]

# Batch Gradient Descent
def logistic_regression(X, y, learning_rate=0.1, epochs=1000):
    m, n = X.shape
    weights = np.zeros((n, 1))
    costs = []
    for i in range(epochs): # epochs vs cost relationship visualization 
        h = sigmoid(np.dot(X, weights))
        gradient = (1/m) * np.dot(X.T, (h - y))
        weights -= learning_rate * gradient
        if i % 10 == 0:# ====> plot every 10 epochs
            costs.append(compute_cost(X, y, weights))
    plt.plot(range(0, epochs, 10), costs)
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.title("Cost vs. Epochs")
    plt.show()
    return weights

# Train the model
weights = logistic_regression(X_train, y_train)

# Predict on training data
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
