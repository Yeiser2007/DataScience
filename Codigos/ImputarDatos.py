import pandas as pd
from sklearn.impute import KNNImputer
# Cargar el dataset desde el archivo CSV
file_path = 'dataset3.csv'
df = pd.read_csv(file_path)

# Mostrar las primeras filas para verificar la carga correcta
print(df.head())

# Verificar si hay valores faltantes en el dataset
print(df.isnull().sum())
# Crear el imputador KNN
imputer = KNNImputer(n_neighbors=5)  # Puedes ajustar el número de vecinos (n_neighbors) según sea necesario

# Definir las columnas numéricas para imputar (excluyendo las primeras dos columnas)
columns_to_impute = df.columns[:]  # Empieza desde la tercera columna hasta el final

# Aplicar la imputación solo a las columnas numéricas
df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])

# Mostrar el DataFrame con los datos imputados
print(df.head())

# Verificar si quedan valores faltantes después de la imputación
print(df.isnull().sum())
# Guardar el DataFrame con los datos imputados en un nuevo archivo CSV
df.to_csv('datos_imputados.csv', index=False)
