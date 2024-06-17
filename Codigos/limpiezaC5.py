import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

# Cargar los datos
data = pd.read_csv('Conjuntos/dataset5.csv')

# Eliminar columnas innecesarias
columns_to_drop = ['periodo_denuncia', 'anio_denuncia', 'fecha_descarga', 'ubigeo_pjfs',
                   'dpto_pjfs', 'prov_pjfs', 'dist_pjfs', 'fecha_corte']
data = data.drop(columns=columns_to_drop)

# Reemplazar 'S/art' en la columna 'articulo' con NaN
data['articulo'] = pd.to_numeric(data['articulo'].replace('S/art', pd.NA), errors='coerce')

# Convertir datos categóricos a numéricos, asegurándonos de no tener valores 0
label_encoders = {}
categorical_columns = ['distrito_fiscal', 'especialidad', 'tipo_caso', 'generico', 'subgenerico', 'des_articulo']

for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column].astype(str)) + 1  # Agregar 1 para evitar valores 0

# Imputación con KNN para datos numéricos
knn_imputer = KNNImputer(n_neighbors=5)
data_imputed = knn_imputer.fit_transform(data)

# Convertir el array de nuevo a un DataFrame
data_imputed = pd.DataFrame(data_imputed, columns=data.columns)

# Volver a convertir las columnas categóricas a su formato original, restando 1 para revertir el ajuste anterior
for column in categorical_columns:
    data_imputed[column] = label_encoders[column].inverse_transform((data_imputed[column].astype(int)) - 1)

# Guardar el archivo procesado
data_imputed.to_csv('datos_procesados_knn.csv', index=False)

# Revisar las clases únicas en cada columna categórica
for column in categorical_columns:
    unique_classes = data_imputed[column].unique()
    print(f"Clases en la columna {column}: {unique_classes}")
