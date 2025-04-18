import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump

# === Load training dataset ===
train_df = pd.read_csv('training_shooter_game_cheater_dataset_500.csv')
X_train = train_df.drop(columns=['Cheater']).values
y_train = train_df['Cheater'].values

# === Load custom testing dataset ===
test_df = pd.read_csv('shooter_game_test_40.csv')  # your custom test file
X_test = test_df.drop(columns=['Cheater']).values
y_test = test_df['Cheater'].values

# === Normalize data based on training set ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Train Ridge-Regularized Logistic Regression ===
model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)
model.fit(X_train_scaled, y_train)

# === Predict on test set ===
y_pred_test = model.predict(X_test_scaled)
y_probs = model.predict_proba(X_test_scaled)[:, 1]  # probabilities for class "1" (cheater)

# === Evaluate performance ===
train_accuracy = accuracy_score(y_train, model.predict(X_train_scaled))
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print("\nConfusion Matrix (Test):")
cm = confusion_matrix(y_test, y_pred_test)
print(cm)

print("\nClassification Report (Test):")
print(classification_report(y_test, y_pred_test))

# ===  Confusion Matrix ===
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# === Pairplot of test data ===
df = test_df.copy()
df['Cheater'] = df['Cheater'].astype(str)  # Convert for hue
sns.pairplot(df, hue='Cheater', palette='Set1', diag_kind='hist', corner=True)
plt.suptitle('Feature Distributions by Cheater Status', y=1.02)
plt.show()

# === Histogram of prediction probabilities ===
plt.hist(y_probs, bins=50, edgecolor='black')
plt.title('Prediction Probability Distribution')
plt.xlabel('Predicted Probability of Being a Cheater')
plt.ylabel('Frequency')
plt.show()

# === Save model and scaler ===
dump(model, 'ridge_logistic_model.joblib')
dump(scaler, 'scaler.joblib')

print("Training complete. Model and scaler saved.")
