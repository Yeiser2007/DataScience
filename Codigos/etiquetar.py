import pandas as pd

# Cargar el archivo CSV
df = pd.read_csv('Datasetcomplete.csv')

# Mapear los valores en la columna del_clas a las etiquetas deseadas
mapeo = {1: 'violencia', 2: 'afectacion_economica', 3: 'entorno_urbano'}
df['etiqueta'] = df['del_clas'].map(mapeo)

# Mover la columna etiqueta al principio del DataFrame
columnas = ['etiqueta'] + [col for col in df.columns if col != 'etiqueta']
df = df[columnas]

# Eliminar las columnas fecha_hecho y fecha_inicio
df = df.drop(['fecha_hecho', 'fecha_inicio'], axis=1)

# Guardar el DataFrame modificado en un nuevo archivo CSV
df.to_csv('etiquetado.csv', index=False)
