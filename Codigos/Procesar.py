import pandas as pd

# Leer el dataset desde el archivo CSV
df = pd.read_csv('dataset_limpio.csv')

# Eliminar las columnas especificadas
df = df.drop(columns=['edad', 'diferencia_fecha', 'del_clas'])

# Guardar el nuevo dataset en un archivo CSV
df.to_csv('ruta/al/nuevo_archivo.csv', index=False)

# Imprimir las primeras filas del nuevo dataset para verificar
print(df.head())
