import pandas as pd
from tqdm import tqdm

# Cargar el archivo CSV
df = pd.read_csv('../Conjuntos/dataset3.csv')

# Eliminar datos menores a 2019-01-01 en la columna fecha_hecho
total_rows = len(df)
indices_a_eliminar = []

for index, row in tqdm(df.iterrows(), total=total_rows, desc="Procesando"):
    fecha_hecho = pd.to_datetime(row['año'], errors='coerce')
    if fecha_hecho < pd.to_datetime('2024'):
        indices_a_eliminar.append(index)

df = df.drop(indices_a_eliminar)

# Guardar el DataFrame resultante en un nuevo archivo CSV
df.to_csv('nuevo_archivo.csv', index=False)

# import pandas as pd
# from tqdm import tqdm

# # Cargar el archivo CSV
# df = pd.read_csv('dataset.csv')

# # Filtrar los registros en donde categoria_delito no sea 'DELITO DE BAJO IMPACTO' o 'HECHO NO DELICTIVO'
# mask = (df['categoria_delito'] != 'DELITO DE BAJO IMPACTO') & (df['categoria_delito'] != 'HECHO NO DELICTIVO')
# df = df[mask]

# # Eliminar datos menores a 2019-01-01 en la columna fecha_hecho
# fecha_hecho_columna = pd.to_datetime(df['fecha_hecho'], errors='coerce')
# df = df[fecha_hecho_columna >= pd.to_datetime('2019-01-01')]

# # Eliminar registros con más de 3 valores vacíos
# df = df.dropna(thresh=df.shape[1]-3)

# # Mostrar la barra de progreso mientras se procesan los datos
# tqdm.pandas(desc="Procesando")

# # Guardar el DataFrame resultante en un nuevo archivo CSV
# df.to_csv('nv.csv', index=False)



# import pandas as pd

# # Cargar el archivo CSV
# df = pd.read_csv('casi_limpp.csv')

# # Eliminar las columnas especificadas
# columnas_a_eliminar = ['hora_hecho','hora_inicio',]
# df = df.drop(columns=columnas_a_eliminar)

# # Guardar el DataFrame resultante en un nuevo archivo CSV
# df.to_csv('casi.csv', index=False)





# import pandas as pd

# # Cargar el archivo CSV
# df = pd.read_csv('date.csv')

# # Separar la columna fecha_hecho en año_hecho, mes_hecho y dia_hecho
# df['fecha_hecho'] = pd.to_datetime(df['fecha_hecho'])
# df['anio_hecho'] = df['fecha_hecho'].dt.year
# df['mes_hecho'] = df['fecha_hecho'].dt.month
# df['dia_hecho'] = df['fecha_hecho'].dt.day

# # Eliminar la columna original fecha_hecho si lo deseas
# # df.drop(columns=['fecha_hecho'], inplace=True)

# # Guardar el DataFrame resultante en un nuevo archivo CSV
# df.to_csv('date2.csv', index=False)



#CLASIFICAR HORA DIA :   1 = {6:00am a 11:59am}, 2= {12:00 pm a 17:59 pm},3 = {6:00pm a 11:59pm}, 4 ={ 12:00am a 5:59}


# import pandas as pd
# from tqdm import tqdm

# # Cargar el archivo CSV
# df = pd.read_csv('casi1.csv')

# # Crear función para clasificar las horas del día
# def clasificar_hora(hora):
#     if hora >= 6 and hora < 12:
#         return 1
#     elif hora >= 12 and hora < 18:
#         return 2
#     elif hora >= 18 and hora < 24:
#         return 3
#     else:
#         return 4

# # Convertir la columna 'hora_hecho' a tipo datetime
# df['hora_hecho'] = pd.to_datetime(df['hora_hecho']).dt.hour

# # Mostrar la barra de progreso mientras se procesan los datos
# tqdm.pandas(desc="Procesando")

# # Clasificar las horas del día
# df['hora_clasificada'] = df['hora_hecho'].progress_apply(clasificar_hora)

# # Guardar el DataFrame resultante en un nuevo archivo CSV
# df.to_csv('hora_clas.csv', index=False)

# import pandas as pd
# import numpy as np

# # Cargar el archivo CSV
# df = pd.read_csv('hora_clas.csv')

# # Reemplazar todos los valores vacíos por NaN
# df = df.replace(r'^\s*$', np.nan, regex=True)

# # Guardar el resultado en un nuevo archivo CSV
# df.to_csv('nan.csv', index=False)


# import pandas as pd
# from tqdm import tqdm

# # Diccionario de mapeo de identificadores de alcaldías a nombres
# mapeo_alcaldias = {
#     'LA MAGDALENA CONTRERAS': '008',
#     'CUAUHTEMOC': '015',
#     'IZTAPALAPA': '007',
#     'MIGUEL HIDALGO': '016',
#     'TLAHUAC': '011',
#     'IZTACALCO': '006',
#     'GUSTAVO A. MADERO': '005',
#     'VENUSTIANO CARRANZA': '017',
#     'BENITO JUAREZ': '014',
#     'ALVARO OBREGON': '010',
#     'TLALPAN': '012',
#     'XOCHIMILCO': '013',
#     'MILPA ALTA': '009',
#     'COYOACAN': '003',
#     'AZCAPOTZALCO': '002',
#     'CUAJIMALPA DE MORELOS': '004'
# }
# # Cargar el archivo CSV
# df = pd.read_csv('hora_clas.csv')

# # Mostrar la barra de progreso mientras se procesan los datos
# tqdm.pandas(desc="Procesando")

# # Clasificar la columna alcaldia_hecho
# df['alcaldia_hecho'] = df['alcaldia_hecho'].progress_apply(lambda x: mapeo_alcaldias.get(x.strip(), x))

# # Guardar el DataFrame resultante en un nuevo archivo CSV
# df.to_csv('alcaldia_clas2.csv', index=False)

# sexo:
# import pandas as pd

# Diccionario de mapeo de delitos a categorías
# import pandas as pd
# mapeo_delitos = {
#     'ROBO DE VEHÍCULO CON Y SIN VIOLENCIA': 21,
#     'LESIONES DOLOSAS POR DISPARO DE ARMA DE FUEGO': 12,
#     'ROBO A NEGOCIO CON VIOLENCIA':22,
#     'ROBO A PASAJERO A BORDO DEL METRO CON Y SIN VIOLENCIA': 31,
#     'ROBO A PASAJERO A BORDO DE MICROBUS CON Y SIN VIOLENCIA': 32,
#     'ROBO A TRANSEUNTE EN VÍA PÚBLICA CON Y SIN VIOLENCIA': 23,
#     'ROBO A PASAJERO A BORDO DE TAXI CON VIOLENCIA': 33,
#     'HOMICIDIO DOLOSO': 11,
#     'ROBO A REPARTIDOR CON Y SIN VIOLENCIA': 34,
#     'ROBO A CASA HABITACIÓN CON VIOLENCIA': 24,
#     'ROBO A CUENTAHABIENTE SALIENDO DEL CAJERO CON VIOLENCIA': 35,
#     'ROBO A TRANSPORTISTA CON Y SIN VIOLENCIA': 25,
#     'VIOLACIÓN': 13,
#     'SECUESTRO': 14
# }

# # Convertir a mayúsculas y eliminar duplicados
# df = pd.read_csv('delitos_clasificados.csv')
# # Reemplazar los valores en la columna 'delito' con sus correspondientes categorías
# df['delito'] = df['categoria_delito'].replace(mapeo_delitos)


# df.to_csv('delitosF.csv', index=False)
# # Guardar el resultado en un archivo CSV




# import pandas as pd

# # Cargar el archivo CSV
# df = pd.read_csv('delitosF.csv')

# # Mapear los valores de género
# mapeo_genero = {'Masculino': 1, 'Femenino': 2}

# # Aplicar el mapeo y reemplazar valores vacíos con NaN
# df['sexo'] = df['sexo'].map(mapeo_genero).fillna(pd.NA)

# # Guardar el resultado en un nuevo archivo CSV
# df.to_csv('casi1.csv', index=False)



# import pandas as pd

# # Diccionario de mapeo de delitos a clasificación
# mapeo_delitos = {
#     'HOMICIDIO DOLOSO': 1,
#     'LESIONES DOLOSAS POR DISPARO DE ARMA DE FUEGO': 1,
#     'FEMINICIDIO': 1,
#     'VIOLACIÓN': 1,
#     'SECUESTRO': 1,
#     'EXTORSIÓN': 2,
#     'ROBO DE VEHÍCULO CON VIOLENCIA': 2,
#     'ROBO DE VEHÍCULO SIN VIOLENCIA': 2,
#     'ROBO A CASA HABITACIÓN CON VIOLENCIA': 2,
#     'ROBO A CASA HABITACIÓN SIN VIOLENCIA': 2,
#     'ROBO A NEGOCIO CON VIOLENCIA': 2,
#     'ROBO A TRANSPORTE PÚBLICO COLECTIVO': 2,
#     'ROBO A TRANSEÚNTE EN VÍA PÚBLICA CON VIOLENCIA': 2,
#     'ROBO A TRANSEÚNTE EN VÍA PÚBLICA SIN VIOLENCIA': 2,
#     'ROBO A TRANSPORTISTA CON VIOLENCIA': 2,
#     'ROBO A TRANSPORTISTA SIN VIOLENCIA': 2,
#     'ROBO A PASAJERO A BORDO DEL METRO CON VIOLENCIA': 3,
#     'ROBO A PASAJERO A BORDO DEL METRO SIN VIOLENCIA': 3,
#     'ROBO A PASAJERO A BORDO DEL METROBÚS CON VIOLENCIA': 3,
#     'ROBO A PASAJERO A BORDO DEL METROBÚS SIN VIOLENCIA': 3,
#     'ROBO A PASAJERO A BORDO DE PESERO COLECTIVO CON VIOLENCIA': 3,
#     'ROBO A PASAJERO A BORDO DE PESERO COLECTIVO SIN VIOLENCIA': 3,
#     'ROBO A PASAJERO A BORDO DE TAXI CON VIOLENCIA': 3,
#     'ROBO A REPARTIDOR CON VIOLENCIA': 3,
#     'ROBO A REPARTIDOR SIN VIOLENCIA': 3,
#     'ROBO A CUENTAHABIENTE SALIENDO DEL CAJERO CON VIOLENCIA': 3
# }

# # Registros de delitos
# delitos = [
#     'ROBO A TRANSEUNTE EN VÍA PÚBLICA CON Y SIN VIOLENCIA',
#     'HOMICIDIO DOLOSO',
#     'ROBO A NEGOCIO CON VIOLENCIA',
#     'ROBO A CASA HABITACIÓN CON VIOLENCIA',
#     'ROBO A PASAJERO A BORDO DE MICROBUS CON Y SIN VIOLENCIA',
#     'LESIONES DOLOSAS POR DISPARO DE ARMA DE FUEGO',
#     'ROBO DE VEHÍCULO CON Y SIN VIOLENCIA',
#     'ROBO A PASAJERO A BORDO DEL METRO CON Y SIN VIOLENCIA',
#     'ROBO A REPARTIDOR CON Y SIN VIOLENCIA',
#     'ROBO A PASAJERO A BORDO DE TAXI CON VIOLENCIA',
#     'ROBO A CUENTAHABIENTE SALIENDO DEL CAJERO CON VIOLENCIA',
#     'ROBO A TRANSPORTISTA CON Y SIN VIOLENCIA',
#     'PLAGIO O SECUESTRO',
#     'FEMINICIDIO',
#     'VIOLACIÓN',
#     'SECUESTRO'
# ]

# # Crear DataFrame con los registros de delitos
# df = pd.DataFrame(delitos, columns=['delito'])

# # Convertir los delitos a mayúsculas
# df['delito'] = df['delito'].str.upper()

# # Calcular el número de coincidencias para cada clasificación
# df['clasificacion_delito'] = df['delito'].apply(lambda x: max(mapeo_delitos.get(y, 0) for y in x.split()))

# # print(df)
# import pandas as pd

# # Cargar el archivo CSV
# df = pd.read_csv('nv.csv')

# # Convertir las columnas de fecha_inicio y fecha_hecho a tipo datetime
# df['fecha_inicio'] = pd.to_datetime(df['fecha_inicio'])
# df['fecha_hecho'] = pd.to_datetime(df['fecha_hecho'])

# # Calcular la diferencia en días entre fecha_hecho y fecha_inicio
# df['diferencia_fecha'] = ((df['fecha_inicio']-df['fecha_hecho']).dt.days)*24

# # Guardar el resultado en un nuevo archivo CSV
# df.to_csv('date_dif.csv', index=False)

# import pandas as pd

# # Cargar el archivo CSV
# df = pd.read_csv('delitos_clasificados.csv')

# # Eliminar las columnas especificadas
# columnas_a_eliminar = ['fecha_inicio','fecha_hecho']
# df = df.drop(columns=columnas_a_eliminar, axis=1)

# # Guardar el resultado en un nuevo archivo CSV
# df.to_csv('datos_listos.csv', index=False)


# DAtos vacios y datos no vacios: import pandas as pd

# Leer el archivo CSV
# import pandas as pd
# df = pd.read_csv('delitos_clasificados.csv')

# # Filtrar los registros con al menos un dato vacío
# registros_con_vacios = df[df.isnull().any(axis=1)]

# # Filtrar los registros sin ningún dato vacío
# registros_sin_vacios = df.dropna()

# # Guardar los resultados en dos archivos CSV separados
# registros_con_vacios.to_csv('registros_con_vacios.csv', index=False)
# registros_sin_vacios.to_csv('registros_sin_vacios.csv', index=False)



# import pandas as pd

# # Cargar el archivo CSV
# df = pd.read_csv('datos_imputados.csv')

# # Definir función para clasificar el rango de edad y reemplazar valores vacíos con NaN
# def clasificar_edad(valor):
#     if pd.isna(valor):
#         return pd.NA
#     elif isinstance(valor, int) and valor in range(0, 13):
#         return 1
#     elif isinstance(valor, int) and valor in range(13, 19):
#         return 2
#     elif isinstance(valor, int) and valor in range(19, 65):
#         return 3
#     elif isinstance(valor, int) and valor >= 65:
#         return 4
#     else:
#         return pd.NA

# # Aplicar la función a la columna edad_categorizada
# df['edad_categorizada'] = df['edad'].apply(clasificar_edad)

# # Guardar el DataFrame resultante en un nuevo archivo CSV
# df.to_csv('new_data.csv', index=False)

# import pandas as pd

# # Cargar el archivo CSV
# df = pd.read_csv('Limpieza/new_data.csv')

# # # Definir las categorías
# categorias = {
#     1: (0, 24),
#     2: (24.1, 72),
#     3: (72.1, 168),
#     4: (168.1, float('inf'))  # Utilizamos float('inf') para representar "mayor que 168"
# }

# # Función para clasificar las horas en las categorías definidas
# def clasificar_horas(horas):
#     for categoria, (limite_inferior, limite_superior) in categorias.items():
#         if limite_inferior <= horas <= limite_superior:
#             return categoria
#     return None  # Devolver None si las horas no se encuentran en ninguna categoría

# # Crear la columna "tiempo-respuesta" clasificando las horas
# df['tiempo-respuesta'] = df['diferencia_fecha'].apply(clasificar_horas)

# # Guardar el resultado en un nuevo archivo CSV
# df.to_csv('Datasetcomplete.csv', index=False)

# import pandas as pd

# def identificar_nan(dataframe):
#     # Identificar valores NaN en el DataFrame
#     nan_values = dataframe.isnull()
    
#     # Contar los valores NaN en cada columna
#     count_nan = nan_values.sum()
    
#     # Obtener un resumen de las columnas con valores NaN
#     columns_with_nan = count_nan[count_nan > 0]
    
#     return nan_values, columns_with_nan

# if __name__ == "__main__":
#     # Nombre del archivo CSV
#     archivo_csv = "etiquetado.csv"  # Cambia esto al nombre de tu archivo CSV
    
#     # Leer el archivo CSV en un DataFrame de pandas
#     df = pd.read_csv(archivo_csv)
    
#     # Identificar y contar los valores NaN
#     valores_nan, columnas_con_nan = identificar_nan(df)
    
#     # Imprimir los resultados
#     print("Valores NaN identificados:")
#     print(valores_nan)
    
#     print("\nColumnas con valores NaN:")
#     print(columnas_con_nan)


# import pandas as pd

# # Cargar el archivo CSV
# df = pd.read_csv('Limpieza/data.csv')

# # Eliminar filas con valores negativos o vacíos en cualquier columna
# df = df[(df >= 0).all(axis=1)].dropna()

# # Guardar los cambios en un nuevo archivo CSV
# df.to_csv('dataset_limpio.csv', index=False)

# # Mostrar el DataFrame actualizado
# print(df)
