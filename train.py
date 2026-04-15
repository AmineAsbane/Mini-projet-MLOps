import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Succès de l'entraînement. Accuracy: {accuracy * 100:.2f}%")

# On s'assure que le dossier 'app' existe pour l'API plus tard
if not os.path.exists('app'):
    os.makedirs('app')

with open('app/model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Modèle sauvegardé dans app/model.pkl")
#Petit comm
