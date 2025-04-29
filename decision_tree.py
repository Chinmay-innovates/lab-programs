import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load the iris dataset
iris = load_iris()

# Create DataFrame with categorical species
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['Species'] = pd.Categorical.from_codes(
    iris.target, categories=iris.target_names)

# Define features and labels
X = df.drop(columns=['Species'])
y = df['Species']
feature_names = X.columns
target_labels = df['Species'].cat.categories  # Ensure correct order of labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Train Decision Tree
dtc = DecisionTreeClassifier(max_depth=3, random_state=42)
dtc.fit(X_train, y_train)

# Plot Decision Tree
plt.figure(figsize=(12, 8), facecolor='lightgray')
plot_tree(dtc, feature_names=feature_names, class_names=target_labels,
          filled=True, rounded=True, fontsize=12)
plt.show()

# Predict and evaluate
y_pred = dtc.predict(X_test)
conf_matrix = metrics.confusion_matrix(y_test, y_pred)

# Plot Confusion Matrix
fig, axis = plt.subplots(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='g',
            cmap='coolwarm', cbar=False, ax=axis)

# Set labels and tick formatting
axis.set_title('Confusion Matrix', fontsize=16)
axis.set_xlabel('Predicted Label', fontsize=14)
axis.set_ylabel('True Label', fontsize=14)
axis.set_xticklabels(target_labels, fontsize=12)
axis.set_yticklabels(target_labels, fontsize=12)

plt.show()

# Print classification report
print(metrics.classification_report(y_test, y_pred))
