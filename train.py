import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import joblib

# 1. Cargar el dataset
iris = load_iris()
X, y = iris.data, iris.target

# 2. Separar datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Entrenar el modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluar el modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 5. Guardar las métricas en un archivo
with open("metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.2f}\n")

# 6. Guardar el modelo entrenado
joblib.dump(model, 'model.pkl')

print("Modelo y métricas guardados.")