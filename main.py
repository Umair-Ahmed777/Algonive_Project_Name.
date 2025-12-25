"""
Employee Attrition Prediction System
Key Features:
- HR Analytics Dataset
- Feature Engineering
- Classification Models
- Evaluation Metrics
- Visualization
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("data/employee_attrition.csv")

# Encode target
le = LabelEncoder()
df["Attrition"] = le.fit_transform(df["Attrition"])

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

sns.countplot(x=df["Attrition"])
plt.title("Employee Attrition Distribution")
plt.show()