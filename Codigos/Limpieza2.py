import pandas as pd
from tqdm import tqdm

# Cargar el archivo CSV especificando 'latin1' como encoding
df = pd.read_csv('datasets/dataset3.csv', encoding='latin1')

# # Eliminar datos con año menor a 2019 en la columna 'año'
# indices_a_eliminar = []

# for index, row in tqdm(df.iterrows(), total=len(df), desc="Procesando"):
#     año = int(row['Año'])  # Convertir el valor de la columna 'año' a entero
#     if año < 2023:
#         indices_a_eliminar.append(index)

# df = df.drop(indices_a_eliminar)

# # Guardar el DataFrame resultante en un nuevo archivo CSV
# df.to_csv('nuevo_archivo.csv', index=False)

# print("Proceso completado. Se han eliminado las filas con año menor a 2019.")


# # Mapear los valores de género
# mapeo_genero = {'Hombre': 1, 'Mujer': 2}

# # Aplicar el mapeo y reemplazar valores vacíos con NaN
# df['Sexo'] = df['Sexo'].map(mapeo_genero).fillna(pd.NA)

# Eliminar columnas específicas
columnas_a_eliminar = ['Enero','Febrero','Marzo','Abril','Mayo','Junio','Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre']
df = df.drop(columns=columnas_a_eliminar)

# # Guardar el DataFrame resultante en un nuevo archivo CSV
# df.to_csv('nuevo_archivo2.csv', index=False)

# print("Proceso completado. Se han eliminado las columnas especificadas y clasificado las entidades según el número de estado.")



# clasificacion_delitos = {
#     'Homicidio': 1,  # Delitos contra la Vida y la Integridad Corporal
#     'Lesiones': 1,  # Delitos contra la Vida y la Integridad Corporal
#     'Feminicidio': 1,  # Delitos contra la Vida y la Integridad Corporal
#     'Otros delitos que atentan contra la vida y la integridad corporal': 1,  # Delitos contra la Vida y la Integridad Corporal
#     'Aborto': 1,  # Delitos contra la Vida y la Integridad Corporal
#     'Secuestro': 2,  # Delitos contra la Libertad y Seguridad
#     'Tráfico de menores': 2,  # Delitos contra la Libertad y Seguridad
#     'Rapto': 2,  # Delitos contra la Libertad y Seguridad
#     'Otros delitos que atentan contra la libertad personal': 2,  # Delitos contra la Libertad y Seguridad
#     'Extorsión': 2,  # Delitos contra la Libertad y Seguridad
#     'Trata de personas': 2,  # Delitos contra la Libertad y Seguridad
#     'Corrupción de menores': 3,  # Delitos contra la Familia y los Menores
#     'Otros delitos contra la sociedad': 4  # Delitos contra el Orden Público y la Sociedad
# }
# # Aplicar la clasificación a una nueva columna 'Categoria del delito'
# df['Categoria del delito'] = df['Tipo de delito'].map(clasificacion_delitos)
# # Guardar el DataFrame resultante en un nuevo archivo CSV
# df.to_csv('del_class.csv', index=False)

# clasificacion_modalidad = {
#     'Con arma de fuego': 1,
#     'Con arma blanca': 1,
#     'Con otro elemento': 1,
#     'No especificado': 1,
#     'En accidente de tránsito': 1,
#     'Secuestro extorsivo': 2,
#     'Secuestro con calidad de rehén': 2,
#     'Secuestro para causar daño': 2,
#     'Secuestro exprés': 2,
#     'Otro tipo de secuestros': 2
# }
# # Aplicar la clasificación a una nueva columna 'Categoria del delito'
# df['Modalidad'] = df['Modalidad'].map(clasificacion_modalidad)

# # Clasificación de edades
# clasificacion_edades = {
#     'Menores de edad (0-17)': 1,
#     'Adultos (18 y más)': 2,
#     'No especificado': pd.NA,
#     'No identificado': pd.NA
# }

# # Aplicar la clasificación de edades
# # df['Clasificacion de edad'] = df['Rango de edad'].map(clasificacion_edades)
# # Lista de columnas en el orden deseado
# columnas_ordenadas = ['Anio', 'Tipo de delito', 'Categoria del delito', 'Clasificacion de edad', 
#                       'Modalidad', 'Sexo', 'Rango de edad', 'Enero', 'Febrero', 'Marzo', 'Abril', 
#                       'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']

# # Reordenar las columnas
# # df = df[columnas_ordenadas]

# orden_columnas = ['Categoria del delito','Clasificacion de edad', 'Clave_Ent', 'Modalidad', 'Sexo', 'Enero', 'Febrero', 
#                   'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 
#                   'Octubre', 'Noviembre', 'Diciembre' ]

# # Reordenar las columnas
# df = df[orden_columnas]

# Guardar el DataFrame resultante en un nuevo archivo CSV
df.to_csv('dataset3.csv', index=False)