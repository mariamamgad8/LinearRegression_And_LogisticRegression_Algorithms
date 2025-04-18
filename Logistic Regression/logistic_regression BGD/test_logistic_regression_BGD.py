import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
#-----------BGD-----------

# Load test data
test_df = pd.read_csv('shooter_game_test_40.csv')
X_test = test_df.drop(columns=['Cheater']).values
y_test = test_df['Cheater'].values

# Load the saved model and scaler
model = joblib.load('logistic_model.pkl')
scaler = joblib.load('scaler.pkl')

# Normalize test data using the loaded scaler
X_test_scaled = scaler.transform(X_test)

# Predict probabilities and labels
y_probs = model.predict_proba(X_test_scaled)[:, 1]  # Probabilities for class "1" (cheater)
y_pred = model.predict(X_test_scaled)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Optional: Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Cheater", "Cheater"]))

# Optional: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Cheater", "Cheater"])
disp.plot(cmap='Reds')
plt.title("Confusion Matrix")
plt.show()

# Optional: Feature distribution visualization
df = test_df.copy()
df['Cheater'] = df['Cheater'].astype(str)
sns.pairplot(df, hue='Cheater', palette='Set1', diag_kind='hist', corner=True)
plt.suptitle('Feature Distributions by Cheater Status', y=1.02)
plt.show()

# Optional: Prediction Probability Histogram
plt.hist(y_probs, bins=50, alpha=0.7, label='Probabilities')
plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold = 0.5')
plt.title('Prediction Probability Distribution')
plt.xlabel('Predicted Probability (Cheater)')
plt.ylabel('Frequency')
plt.legend()
plt.show()







