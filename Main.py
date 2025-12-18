# main.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('dataset.csv')
df.info()
df.describe()

corrmat = df.corr()
sns.heatmap(corrmat, annot=True, cmap="RdYlGn", square=True)
sns.countplot(x='target', data=df, palette='RdBu_r')

dataset = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
scaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = scaler.fit_transform(dataset[columns_to_scale])

X = dataset.drop(['target'], axis=1)
y = dataset['target']

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

knn_scores = []  # ✅ Initialize the list

for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X, y, cv=10)
    knn_scores.append(score.mean())

# Plotting the scores
import matplotlib.pyplot as plt

plt.plot(range(1, 21), knn_scores, color='red')
for i in range(1, 21):
    plt.text(i, knn_scores[i-1], f"{knn_scores[i-1]:.2f}")
plt.xticks(range(1, 21))
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Cross-Validation Score')
plt.title('KNN Scores for Different K Values')
plt.show()

rf = RandomForestClassifier(n_estimators=10)
score = cross_val_score(rf, X, y, cv=10)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train best model (example with KNN)
best_model = KNeighborsClassifier(n_neighbors=12)
best_model.fit(X_train, y_train)

# Predictions
y_pred = best_model.predict(X_test)

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

import joblib

# Train your final model
final_model = KNeighborsClassifier(n_neighbors=12)
final_model.fit(X, y)

# Save model and feature list
joblib.dump(final_model, 'heart_disease_model.pkl')
joblib.dump(list(X.columns), 'model_features.pkl')
