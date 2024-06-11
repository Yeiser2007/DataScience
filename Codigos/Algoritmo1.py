import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from imblearn.metrics import geometric_mean_score
import numpy as np

# Supongamos que el dataset está en un archivo CSV llamado "dataset.csv"
data = pd.read_csv('dataset.csv')

# Separar las características (X) y la etiqueta (y)
X = data.drop('etiqueta', axis=1)
y = data['etiqueta']

# Definir la validación cruzada con 10 particiones
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Inicializar listas para almacenar las métricas de cada iteración
accuracies = []
geometric_means = []

# Iterar sobre cada partición
for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
    # Dividir los datos en entrenamiento y prueba
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Guardar los datos de entrenamiento en un CSV
    train_data = X_train.copy()
    train_data['etiqueta'] = y_train
    train_data.to_csv(f'train_data_fold_{fold+1}.csv', index=False)
    
    # Entrenar el modelo KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    
    # Hacer predicciones
    y_pred = knn.predict(X_test)
    
    # Calcular precisión y media geométrica
    accuracy = accuracy_score(y_test, y_pred)
    geometric_mean = geometric_mean_score(y_test, y_pred, average='macro')
    
    # Almacenar las métricas
    accuracies.append(accuracy)
    geometric_means.append(geometric_mean)

# Calcular las estadísticas finales
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
mean_geometric_mean = np.mean(geometric_means)
std_geometric_mean = np.std(geometric_means)

# Mostrar los resultados
print(f'Mean Accuracy: {mean_accuracy:.4f}')
print(f'Standard Deviation of Accuracy: {std_accuracy:.4f}')
print(f'Mean Geometric Mean: {mean_geometric_mean:.4f}')
print(f'Standard Deviation of Geometric Mean: {std_geometric_mean:.4f}')
