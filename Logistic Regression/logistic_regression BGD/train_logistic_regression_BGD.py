import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
#-----------BGD-----------

# Load training data
train_df = pd.read_csv('training_shooter_game_cheater_dataset_500.csv')

# Separate features and label
X_train = train_df.drop(columns=['Cheater']).values
y_train = train_df['Cheater'].values

# Normalize features using StandardScaler (standardization: mean=0, std=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Predict on training data
y_pred_train = model.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, y_pred_train)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

# Confusion matrix
cm = confusion_matrix(y_train, y_pred_train)
print("Confusion Matrix:")
print(cm)

# Save model and scaler
joblib.dump(model, 'logistic_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Training complete. Saved: logistic_model.pkl, scaler.pkl.")
