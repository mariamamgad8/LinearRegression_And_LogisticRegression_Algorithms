import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#-----------SGD from scratch-----------
# Load training data
train_df = pd.read_csv('training_shooter_game_cheater_dataset_500.csv')
X_train = train_df.drop(columns=['Cheater']).values
y_train = train_df['Cheater'].values.reshape(-1, 1)

# Normalize (feature scaling)
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std

# Add bias term
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# SGD for logistic regression
def sgd_logistic_regression(X, y, lr=0.01, epochs=1000):
    m, n = X.shape
    weights = np.zeros((n, 1))

    for epoch in range(epochs):
        for i in range(m):
            xi = X[i].reshape(1, -1)
            yi = y[i]
            pred = sigmoid(np.dot(xi, weights))
            error = pred - yi
            gradient = xi.T * error
            weights -= lr * gradient

        if epoch % 10 == 0:
            loss = -np.mean(yi * np.log(pred + 1e-5) + (1 - yi) * np.log(1 - pred + 1e-5))
            print(f"Epoch {epoch} | Loss: {loss:.4f}")

    return weights

# Train the model
weights = sgd_logistic_regression(X_train, y_train, lr=0.01, epochs=100)

# Save model parameters
np.save('weights.npy', weights)
np.save('mean.npy', mean)
np.save('std.npy', std)

# ============================
# Load and evaluate on test set
# ============================
test_df = pd.read_csv('shooter_game_test_40.csv')
X_test = test_df.drop(columns=['Cheater']).values
y_test = test_df['Cheater'].values.reshape(-1, 1)

# Normalize using training stats and add bias  ============>AKA "scalling"
X_test = (X_test - mean) / std
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

def predict(X, weights):
    return (sigmoid(np.dot(X, weights)) >= 0.5).astype(int)

# Predict
y_pred = predict(X_test, weights)
accuracy = (y_pred == y_test).mean()
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Plot pairplot
df = pd.read_csv('shooter_game_test_40.csv')
df['Cheater'] = df['Cheater'].astype(str)
sns.pairplot(df, hue='Cheater', palette='Set1', diag_kind='hist', corner=True)
plt.suptitle('Feature Distributions by Cheater Status', y=1.02)
plt.show()

# Plot prediction probability distribution
y_probs = sigmoid(np.dot(X_test, weights))
plt.hist(y_probs, bins=50)
plt.title('Prediction Probability Distribution')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.show()
