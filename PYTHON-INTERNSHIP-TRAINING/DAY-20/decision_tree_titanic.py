# Step 1: Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 2: Load data
titanic = sns.load_dataset('titanic')

# Step 3: Look at data
print(titanic.head())
print(titanic.info())

# Step 4: Choose target value (what we want to predict)
# Target = survived

# Step 5: Choose input features (what helps us predict)
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']

# Step 6: Clean data (simple way)
titanic = titanic[features + ['survived']]
titanic.dropna(inplace=True)  # Remove rows with missing values
titanic['sex'] = titanic['sex'].map({'male': 0, 'female': 1})  # Convert sex to numeric

# Step 7: Split data (train & test)
X = titanic[features]
y = titanic['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Build the Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 9: Make prediction
y_pred = model.predict(X_test)

# Step 10: Check how good the model is (calculate accuracy)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Step 11: Visualize the Decision Tree
plt.figure(figsize=(15, 10))
plot_tree(model, feature_names=features, class_names=['Died', 'Survived'], filled=True)
plt.title("Decision Tree - Titanic")
plt.show()

# Step 12: See which features matter most
feature_importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("Feature Importance:\n", feature_importance)
