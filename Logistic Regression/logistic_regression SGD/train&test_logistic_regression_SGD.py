import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# =========================
# Load training data
# =========================
train_df = pd.read_csv('training_shooter_game_cheater_dataset_500.csv')
X_train = train_df.drop(columns=['Cheater']).values
y_train = train_df['Cheater'].values

# =========================
# Normalize (standardize) features
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# =========================
# Train Logistic Regression with SGD
# =========================
model = SGDClassifier(loss='log_loss', learning_rate='optimal', max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(model, 'logistic_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# =========================
# Load and evaluate on test set
# =========================
test_df = pd.read_csv('shooter_game_test_40.csv')
X_test = test_df.drop(columns=['Cheater']).values
y_test = test_df['Cheater'].values

# Normalize test data
X_test_scaled = scaler.transform(X_test)

# Predict
y_pred = model.predict(X_test_scaled)
y_probs = model.predict_proba(X_test_scaled)[:, 1]

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# =========================
# Plot pairplot
# =========================
df = test_df.copy()
df['Cheater'] = df['Cheater'].astype(str)
sns.pairplot(df, hue='Cheater', palette='Set1', diag_kind='hist', corner=True)
plt.suptitle('Feature Distributions by Cheater Status', y=1.02)
plt.show()

# =========================
# Plot prediction probabilities
# =========================
plt.hist(y_probs, bins=50)
plt.title('Prediction Probability Distribution')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.show()
