import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#-----------Loss from scratch-----------
# Load test data
test_df = pd.read_csv('shooter_game_test_40.csv')
X_test = test_df.drop(columns=['Cheater']).values
y_test = test_df['Cheater'].values.reshape(-1, 1)

# Load saved model
weights = np.load('weights.npy')
mean = np.load('mean.npy')
std = np.load('std.npy')

# Normalize and add bias    ============>AKA "scalling"
X_test = (X_test - mean) / std
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# Predict
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, weights):
    return (sigmoid(np.dot(X, weights)) >= 0.5).astype(int)

y_pred = predict(X_test, weights)
accuracy = (y_pred == y_test).mean() # the mean of number of correct predictions
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# data distribution visualization
df = pd.read_csv('shooter_game_test_40.csv')


df['Cheater'] = df['Cheater'].astype(str)

# Create the pairplot
sns.pairplot(df, hue='Cheater', palette='Set1', diag_kind='hist', corner=True)

# Show plot
plt.suptitle('Feature Distributions by Cheater Status', y=1.02)
plt.show()

y_probs = sigmoid(np.dot(X_test, weights))
plt.hist(y_probs, bins=50)
plt.title('Prediction Probability Distribution')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.show()






