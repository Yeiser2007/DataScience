import pandas as pd

# Cargar el dataset desde el archivo CSV
file_path = 'ruta/del/archivo.csv'  # Cambia esto a la ruta correcta de tu archivo CSV
df = pd.read_csv(file_path)

# Imprimir las columnas para verificar que el archivo se ha cargado correctamente
print("Columnas del DataFrame:", df.columns)

# Conservar columnas relevantes
columns_to_keep = ['delito_articulo', 'delito_descripcion', 'tipo', 'codigo_delito', 'vigente']
df = df[columns_to_keep]

# Convertir 'vigente' a numérico
df['vigente'] = df['vigente'].map({'SI': 1, 'NO': 0})

# Convertir 'tipo' a variables dummy
df = pd.get_dummies(df, columns=['tipo'], drop_first=True)

# Crear una nueva columna 'categoria_delito'
def clasificar_delito(descripcion):
    if 'Homicidio' in descripcion:
        return 'Homicidio'
    elif 'Aborto' in descripcion:
        return 'Aborto'
    elif 'Lesiones' in descripcion:
        return 'Lesiones'
    else:
        return 'Otros'

# Verificar si la columna 'delito_descripcion' existe antes de aplicar la función
if 'delito_descripcion' in df.columns:
    df['categoria_delito'] = df['delito_descripcion'].apply(clasificar_delito)
else:
    print("Error: La columna 'delito_descripcion' no existe en el DataFrame.")
    print("Columnas disponibles:", df.columns)
    exit()

# Convertir 'categoria_delito' a variables dummy
df = pd.get_dummies(df, columns=['categoria_delito'])

# Guardar el DataFrame preprocesado en un nuevo archivo CSV
output_path = 'datos_delitos_preprocesados_con_clasificacion.csv'
df.to_csv(output_path, index=False)

print(f"Archivo preprocesado guardado en {output_path}")
