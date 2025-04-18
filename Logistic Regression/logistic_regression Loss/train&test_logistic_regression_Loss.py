import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# ========== Load and preprocess data ==========

# Load the training dataset
train_df = pd.read_csv('training_shooter_game_cheater_dataset_500.csv')

# Separate features and label for training
X_train = train_df.drop(columns=['Cheater']).values
y_train = train_df['Cheater'].values

# Load your own testing dataset
test_df = pd.read_csv('shooter_game_test_40.csv')  # <-- Replace with your actual test dataset

# Separate features and label for testing
X_test = test_df.drop(columns=['Cheater']).values
y_test = test_df['Cheater'].values

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== Train the model ==========

model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_train_scaled, y_train)

# Predict on training and test sets
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)
y_probs = model.predict_proba(X_test_scaled)[:, 1]  # Probabilities for positive class

# Accuracy
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save model and scaler
np.save('weights.npy', model.coef_.flatten())
np.save('intercept.npy', model.intercept_)
np.save('scaler_mean.npy', scaler.mean_)
np.save('scaler_std.npy', scaler.scale_)

print("Training complete. Saved: weights.npy, intercept.npy, scaler_mean.npy, scaler_std.npy.")

# ========== Visualizations ==========

# --- 1. Confusion Matrix ---
sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# --- 2. Pairplot of Test Features by Cheater Status ---
df = test_df.copy()
df['Cheater'] = df['Cheater'].astype(str)  # Convert to string for hue coloring
sns.pairplot(df, hue='Cheater', palette='Set1', diag_kind='hist', corner=True)
plt.suptitle('Feature Distributions by Cheater Status', y=1.02)
plt.show()

# --- 3. Prediction Probability Distribution ---
plt.hist(y_probs, bins=50, color='skyblue', edgecolor='black')
plt.title('Prediction Probability Distribution')
plt.xlabel('Predicted Probability (Cheater)')
plt.ylabel('Frequency')
plt.show()
