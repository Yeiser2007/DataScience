import csv
import math
import random
import operator

def distancia_euclidiana(vector1, vector2):
    if len(vector1) != len(vector2):
        return -1
    diff = 0
    for index in range(len(vector1)):
        diff += math.pow(vector2[index] - vector1[index], 2)
    distance = math.sqrt(diff)
    return distance

def cargar_datos(archivo_nombre, entrenamientoT):
    d_entrenamiento = []
    d_prueba = []
    with open(archivo_nombre, 'r') as csv_ds_file:
        lineas = csv.reader(csv_ds_file)
        encabezado = next(lineas)  # Guardar la primera línea (encabezado)
        data = list(lineas)
        total_registros = len(data)
        for x in range(total_registros - 1):
            if x % 1000 == 0:
                print(f"Leyendo registro {x + 1} de {total_registros}...")  # Mostrar progreso cada 1000 registros
            for y in range(1, len(data[x])):
                data[x][y] = float(data[x][y])
            if random.random() < entrenamientoT:
                d_entrenamiento.append(data[x])
            else:
                d_prueba.append(data[x])
    print("Datos cargados exitosamente.")
    return d_entrenamiento, d_prueba, encabezado

def similitud(elemento_1, elemento_2):
    return distancia_euclidiana(elemento_1[1:], elemento_2[1:])  # Cambiar para omitir la primera columna que es la etiqueta

def vecinos(training_set, test_element, numero_vecinos):
    distancias = [(elemento, similitud(test_element, elemento)) for elemento in random.sample(training_set, min(numero_vecinos * 2, len(training_set)))]
    distancias.sort(key=operator.itemgetter(1))
    vecinos_mas_cercanos = [distancia[0] for distancia in distancias[:numero_vecinos]]
    return vecinos_mas_cercanos

def obtener_respuesta(vecinos_mas_cercanos):
    votos_clases = {}
    for vecino in vecinos_mas_cercanos:
        clase = vecino[0]  # Tomamos la etiqueta del vecino (primera columna)
        if clase in votos_clases:
            votos_clases[clase] += 1
        else:
            votos_clases[clase] = 1
    clase_predominante = max(votos_clases.items(), key=operator.itemgetter(1))[0]
    return clase_predominante

def exactitud(datos_prueba, predicciones):
    correctos = sum(1 for i in range(len(datos_prueba)) if datos_prueba[i][0] == predicciones[i])  # Comparar con la primera columna que es la etiqueta
    return (correctos / float(len(datos_prueba))) * 100.0

def main():
    # Cargar datos
    print("Cargando datos...")
    datos_entrenamiento, datos_prueba, encabezado = cargar_datos('Procesamiento/dataset.csv', 0.80)
    print('Cantidad de datos de entrenamiento:', len(datos_entrenamiento))
    print('Cantidad de datos de prueba:', len(datos_prueba))
    predicciones = []
    k = 1
    print("Generando predicciones...")
    for i, dato_prueba in enumerate(datos_prueba):
        vecinos_cercanos = vecinos(datos_entrenamiento, dato_prueba, k)
        resultado = obtener_respuesta(vecinos_cercanos)
        predicciones.append(resultado)
        print(f"Registro {i + 1} / {len(datos_prueba)}: Etiqueta actual={dato_prueba[0]}, Etiqueta retiquetada={resultado}")
    
    # Guardar los resultados en un nuevo archivo CSV
    with open('dataset_retiquetado.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(encabezado + ['Etiqueta Retiquetada'])
        for i, dato_prueba in enumerate(datos_prueba):
            csvwriter.writerow(dato_prueba + [predicciones[i]])
    print("Nuevo dataset guardado exitosamente.")

    exac = exactitud(datos_prueba, predicciones)
    print('Exactitud del retiquetado:', exac, '%')

if __name__ == "__main__":
    main()
