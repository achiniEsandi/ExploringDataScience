# Imports
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("titanic.csv")
print("Initial Data Info:")
data.info()
print("\nMissing values before preprocessing:")
print(data.isnull().sum())

# Data Cleaning and Feature Engineering
def preprocess_data(df):
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

    df["Embarked"] = df["Embarked"].fillna("S")
    df = df.drop(columns=["Embarked"])

    df["Age"] = df.groupby("Pclass")["Age"].transform(lambda x: x.fillna(x.median()))

    df["Sex"] = df["Sex"].map({'male': 1, 'female': 0})

    df["FamilySize"] = df["SibSp"] + df["Parch"]
    df["IsAlone"] = (df["FamilySize"] == 0).astype(int)

    df["FareBin"] = pd.qcut(df["Fare"], 4, labels=False, duplicates='drop')
    df["AgeBin"] = pd.cut(df["Age"], bins=[0, 12, 20, 40, 60, np.inf], labels=False, right=False)

    # Handle any remaining NaNs that qcut/cut may introduce
    df = df.dropna()

    return df


# Apply preprocessing
data = preprocess_data(data)
print("\nData after preprocessing (first 5 rows):")
print(data.head())
print("\nMissing values after preprocessing:")
print(data.isnull().sum())

# Define features (X) and target (y)
X = data.drop(columns=["Survived"])
y = data["Survived"]

# Check for missing values before scaling
print("\nMissing values in X before scaling:")
print(X.isnull().sum())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter Tuning for KNN
def tune_model(X_train, y_train):
    param_grid = {
        "n_neighbors": range(1, 21),
        "metric": ["euclidean", "manhattan", "minkowski"],
        "weights": ["uniform", "distance"]
    }
    model = KNeighborsClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Tune model
best_model = tune_model(X_train, y_train)
print(f"\nBest KNN Model Parameters: {best_model.get_params()}")

# Evaluation
def evaluate_model(model, X_test, y_test):
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    matrix = confusion_matrix(y_test, prediction)
    return accuracy, matrix

# Evaluate
accuracy, matrix = evaluate_model(best_model, X_test, y_test)
print(f'\nModel Evaluation:')
print(f'Accuracy: {accuracy*100:.2f}%')
print(f'Confusion Matrix:\n{matrix}')

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Not Survived', 'Predicted Survived'],
            yticklabels=['Actual Not Survived', 'Actual Survived'])
plt.title('Confusion Matrix for Titanic Survival Prediction')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()
