import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from imblearn.metrics import geometric_mean_score
from tqdm import tqdm

# Leer el archivo CSV
print("Leyendo el archivo CSV...")
df = pd.read_csv('dataset.csv')
print("Archivo CSV leído con éxito.")

# Separar características y etiquetas
print("Separando características y etiquetas...")

X = df.drop('etiqueta', axis=1)
y = df['etiqueta']
print("Separación completada.")

# Configurar la validación cruzada y el clasificador KNN
n_splits = 10
print(f"Configurando la validación cruzada con {n_splits} particiones...")
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
knn = KNeighborsClassifier(n_neighbors=1)  # Ajustar el número de vecinos según sea necesario
print("Validación cruzada y clasificador KNN configurados.")

# Listas para almacenar las métricas
accuracies = []
geometric_means = []

# Proceso de validación cruzada
print("Iniciando el proceso de validación cruzada...")
for fold, (train_index, test_index) in enumerate(tqdm(skf.split(X, y), total=n_splits, desc="Validación cruzada"), 1):
    print(f"Procesando partición {fold}...")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred) * 100  # Convertir a porcentaje
    geometric_mean = geometric_mean_score(y_test, y_pred, average='macro') * 100  # Convertir a porcentaje

    accuracies.append(accuracy)
    geometric_means.append(geometric_mean)
    print(f"Partición {fold} completada: Precisión = {accuracy:.2f}%, Media geométrica = {geometric_mean:.2f}%")

# Calcular las métricas finales
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
mean_geometric_mean = np.mean(geometric_means)
std_geometric_mean = np.std(geometric_means)

print("\nResultados finales:")
print(f'Precisión media: {mean_accuracy:.2f}%')
print(f'Desviación estándar de la precisión: {std_accuracy:.2f}%')
print(f'Media geométrica media: {mean_geometric_mean:.2f}%')
print(f'Desviación estándar de la media geométrica: {std_geometric_mean:.2f}%')
