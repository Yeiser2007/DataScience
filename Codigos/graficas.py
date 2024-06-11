import pandas as pd
import matplotlib.pyplot as plt

# Leer el archivo CSV
print("Leyendo el archivo CSV...")
df = pd.read_csv('dataset_balanceado.csv')
print("Archivo CSV leído con éxito.")

# Generar histogramas para cada columna
print("Generando histogramas para cada columna...")

# Crear una carpeta para guardar los histogramas
import os
if not os.path.exists('histogramas'):
    os.makedirs('histogramas')

for column in df.columns:
    plt.figure(figsize=(10, 6))
    plt.hist(df[column], bins=30, edgecolor='k', alpha=0.7)
    plt.title(f'Histograma de {column}')
    plt.xlabel(column)
    plt.ylabel('Frecuencia')
    plt.grid(axis='y', alpha=0.75)
    # Guardar el histograma como archivo de imagen
    plt.savefig(f'histogramas/histograma_{column}.png')
    plt.close()
    print(f'Histograma de {column} guardado en "histogramas/histograma_{column}.png"')

print("Todos los histogramas se han generado y guardado con éxito.")
