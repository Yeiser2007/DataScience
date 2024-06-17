
import pandas as pd
from sklearn.impute import SimpleImputer
# Cargar el archivo CSV
df = pd.read_csv('dataProcessed/dataset_procesado.csv')


# # Eliminar la columna "Anio" y las columnas de los meses
# columns_to_drop = ['Anio', 'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
# df = df.drop(columns=columns_to_drop)

# # Filtrar las filas para conservar solo ciertos tipos de delito
# delitos_a_conservar = ['Lesiones', 'Homicidio', 'Feminicidio', 'Secuestro']
# df = df[df['Tipo de delito'].isin(delitos_a_conservar)]

# # Reclasificar los valores en la columna "Rango de edad"
# df['Rango de edad'] = df['Rango de edad'].replace({
#     'Menores de edad (0-17)': 1,
#     'Adultos (18 y mÃÂÃÂÃÂÃÂ¡s)': 2
# })

# # Sustituir "No especificado" con valores nulos
# df = df.replace('No especificado', pd.NA)
# df = df.replace('No identificado', pd.NA)

# # Imputar los datos nulos
# Imputar los datos nulos para columnas categóricas con la moda
# Modificar la primera columna
df['etiqueta'] = 'clase' + df['etiqueta'].astype(str)

# Guardar el DataFrame modificado en un nuevo archivo CSV
df.to_csv('dataset_modificado.csv', index=False)

print(df)
# Guardar el DataFrame modificado en un nuevo archivo CSV
df.to_csv('dataWeka/dataset5c.csv', index=False)

# Mostrar el DataFrame modificado
print(df)
