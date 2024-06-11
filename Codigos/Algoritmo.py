import pandas as pd

# Paso 1: Leer el archivo CSV
df = pd.read_csv('casi.csv')

# Paso 2: Identificar columnas con datos faltantes
columnas_con_faltantes = df.columns[df.isnull().any()]

# Paso 3: Imputar datos faltantes utilizando la media y redondear a enteros
for columna in columnas_con_faltantes:
    media = df[columna].mean()
    df[columna].fillna(media, inplace=True)
    df[columna] = df[columna].round().astype(int)  # Redondear a enteros

# Paso 4: Guardar los datos imputados en un nuevo archivo CSV
df.to_csv('datos_imputados.csv', index=False)
