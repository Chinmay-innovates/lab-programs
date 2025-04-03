import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.datasets import load_diabetes

# Load dataset
df = load_diabetes()
X, y = df.data, df.target

# Convert to binary classification (above or below mean)
y_binary = (y > np.median(y)).astype(int)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # Ensure test data is also scaled!

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Print metrics
print("Accuracy:", accuracy_score(y_test, y_pred)*100)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Scatter plot (not a true decision boundary but a visualization)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test[:, 2], y=X_test[:, 8], hue=y_test, palette={0: 'blue', 1: 'red'})

plt.xlabel('BMI')
plt.ylabel('Age')
plt.title('Logistic Regression Scatter Plot',)
plt.legend(title='Diabetes')

plt.show()
