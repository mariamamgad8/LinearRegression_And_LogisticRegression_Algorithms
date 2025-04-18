import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from main_logistic_regression import LogisticRegression

# Load datasets
train_data = pd.read_csv('training_shooter_game_cheater_dataset_500.csv')
test_data = pd.read_csv('testing_shooter_game_cheater_dataset_500.csv')

# Split features and target
X_train = train_data.drop('Cheater', axis=1).values
y_train = train_data['Cheater'].values

X_test = test_data.drop('Cheater', axis=1).values
y_test = test_data['Cheater'].values

# Accuracy function
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

# Initialize and train model
regressor = LogisticRegression(lr=0.0001, n_iters=15000)
regressor.fit(X_train, y_train)

# Predict on test set
predictions = regressor.predict(X_test)

# Print accuracy
print("Logistic Regression classification accuracy:", round(accuracy(y_test, predictions) * 100, 2), "%")

# --- Visualization ---

# Plot loss curve
try:
    plt.figure(figsize=(10,5))
    plt.plot(regressor.losses)
    plt.title('Training Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
except AttributeError:
    print("Losses are not being tracked in the LogisticRegression class.")

# Plot confusion matrix
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
