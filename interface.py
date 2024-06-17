import os
import pandas as pd
import numpy as np
import csv
import math
import random
import operator
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

# Variables globales para los datos cargados
datos_entrenamiento = []
datos_prueba = []
encabezado = []

# Funciones existentes para generación de muestras sintéticas y KNN
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
        encabezado = next(lineas)
        if encabezado[0].lower() != 'etiqueta':
            encabezado[0] = 'Etiqueta'
        data = list(lineas)
        total_registros = len(data)
        for x in range(total_registros):
            if x % 1000 == 0:
                print(f"Leyendo registro {x + 1} de {total_registros}...")
            for y in range(1, len(data[x])):
                data[x][y] = float(data[x][y])
            if random.random() < entrenamientoT:
                d_entrenamiento.append(data[x])
            else:
                d_prueba.append(data[x])
    print("Datos cargados exitosamente.")
    return d_entrenamiento, d_prueba, encabezado

def similitud(elemento_1, elemento_2):
    return distancia_euclidiana(elemento_1[1:], elemento_2[1:])

def vecinos(training_set, test_element, numero_vecinos):
    distancias = [(elemento, similitud(test_element, elemento)) for elemento in random.sample(training_set, min(numero_vecinos * 2, len(training_set)))]
    distancias.sort(key=operator.itemgetter(1))
    vecinos_mas_cercanos = [distancia[0] for distancia in distancias[:numero_vecinos]]
    return vecinos_mas_cercanos

def obtener_respuesta(vecinos_mas_cercanos):
    votos_clases = {}
    for vecino in vecinos_mas_cercanos:
        clase = vecino[0]
        if clase in votos_clases:
            votos_clases[clase] += 1
        else:
            votos_clases[clase] = 1
    clase_predominante = max(votos_clases.items(), key=operator.itemgetter(1))[0]
    return clase_predominante

def exactitud(datos_prueba, predicciones):
    correctos = sum(1 for i in range(len(datos_prueba)) if datos_prueba[i][0] == predicciones[i])
    return (correctos / float(len(datos_prueba))) * 100.0

def limpiar_interfaz():
    for frame in [frame_dataset, frame_barras, frame_result]:
        for widget in frame.winfo_children():
            widget.destroy()

def cargar_archivo():
    global datos_entrenamiento, datos_prueba, encabezado
    archivo_nombre = filedialog.askopenfilename()
    if archivo_nombre:
        limpiar_interfaz()
        datos_entrenamiento, datos_prueba, encabezado = cargar_datos(archivo_nombre, 0.80)
        print('Cantidad de datos de entrenamiento:', len(datos_entrenamiento))
        print('Cantidad de datos de prueba:', len(datos_prueba))
        mostrar_dataset(datos_entrenamiento, encabezado, frame_dataset)
        mostrar_grafica_barras(datos_entrenamiento, "Original Dataset", frame_barras)

        # Guardar el conjunto de datos cargado en DataInterfaz/folds/
        os.makedirs('DataInterfaz/folds/', exist_ok=True)
        df_entrenamiento = pd.DataFrame(datos_entrenamiento, columns=encabezado)
        df_prueba = pd.DataFrame(datos_prueba, columns=encabezado)
        df_entrenamiento.to_csv(f'DataInterfaz/folds/datos_entrenamiento_{os.path.basename(archivo_nombre)}', index=False)
        df_prueba.to_csv(f'DataInterfaz/folds/datos_prueba_{os.path.basename(archivo_nombre)}', index=False)

def main_knn():
    global datos_entrenamiento, datos_prueba, encabezado
    if datos_entrenamiento and datos_prueba:
        predicciones = []
        k = int(k_var.get())  # Obtener el valor de K ingresado por el usuario
        print("Generando predicciones...")
        progress_var.set(0)
        progress_step = 100 / len(datos_prueba)

        for i, dato_prueba in enumerate(datos_prueba):
            vecinos_cercanos = vecinos(datos_entrenamiento, dato_prueba, k)
            resultado = obtener_respuesta(vecinos_cercanos)
            predicciones.append(resultado)
            print(f"Registro {i + 1} / {len(datos_prueba)}: Etiqueta actual={dato_prueba[0]}, Etiqueta retiquetada={resultado}")
            progress_var.set(progress_var.get() + progress_step)
            root.update_idletasks()

        # Guardar los resultados en un nuevo archivo CSV
        with open('dataset_retiquetado.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(encabezado + ['Etiqueta Retiquetada'])
            for i, dato_prueba in enumerate(datos_prueba):
                csvwriter.writerow(dato_prueba + [predicciones[i]])
        print("Nuevo dataset guardado exitosamente.")

        exac = exactitud(datos_prueba, predicciones)
        print('Exactitud del retiquetado:', exac, '%')
        mostrar_resultados_knn(exac, datos_prueba, predicciones)

        # Validación cruzada con k=10
        X = np.array([x[1:] for x in datos_entrenamiento])
        y = np.array([x[0] for x in datos_entrenamiento])
        knn = KNeighborsClassifier(n_neighbors=k)
        kf = KFold(n_splits=10)
        y_pred = cross_val_predict(knn, X, y, cv=kf)
        cross_val_acc = accuracy_score(y, y_pred)
        print('Exactitud de validación cruzada:', cross_val_acc * 100, '%')

        conf_matrix = confusion_matrix(y, y_pred)
        metrics = precision_recall_fscore_support(y, y_pred, average=None)
        metrics_dict = {label: {'precision': precision, 'recall': recall, 'f1_score': f1} 
                        for label, precision, recall, f1 in zip(np.unique(y), metrics[0], metrics[1], metrics[2])}
        mostrar_resultados(cross_val_acc * 100, conf_matrix, metrics_dict)

def mostrar_resultados_knn(exactitud, datos_prueba, predicciones):
    for widget in frame_result.winfo_children():
        widget.destroy()
    exactitud_label = tk.Label(frame_result, text=f"Exactitud del retiquetado: {exactitud:.2f}%", font=("Arial", 14))
    exactitud_label.pack(pady=10)
    
    tree = ttk.Treeview(frame_result)
    tree.pack(side="left", fill="both", expand=True)
    scrollbar = ttk.Scrollbar(frame_result, orient="vertical", command=tree.yview)
    scrollbar.pack(side="right", fill="y")
    tree.configure(yscrollcommand=scrollbar.set)

    encabezado = ['Actual'] + ['Predicción']
    tree["columns"] = encabezado
    for col in encabezado:
        tree.heading(col, text=col)
        tree.column(col, anchor="w")

    for i, dato in enumerate(datos_prueba):
        tree.insert("", "end", text=str(i), values=(dato[0], predicciones[i]))

def mostrar_dataset(datos, encabezado, frame):
    for widget in frame.winfo_children():
        widget.destroy()
    tree = ttk.Treeview(frame)
    tree.pack(side="left", fill="both", expand=True)
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    scrollbar.pack(side="right", fill="y")
    tree.configure(yscrollcommand=scrollbar.set)
    
    tree["columns"] = encabezado
    for col in encabezado:
        tree.heading(col, text=col)
        tree.column(col, anchor="w")

    for i, row in enumerate(datos):
        tree.insert("", "end", text=str(i), values=row)

def mostrar_grafica_barras(datos, title, frame):
    for widget in frame.winfo_children():
        widget.destroy()
    clases = [dato[0] for dato in datos]
    clases_unicas, conteos = np.unique(clases, return_counts=True)
    
    fig, ax = plt.subplots()
    ax.bar(clases_unicas, conteos)
    ax.set_xlabel('Clases')
    ax.set_ylabel('Frecuencia')
    ax.set_title(title)

    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

def mostrar_resultados(exactitud, conf_matrix, metrics):
    for widget in frame_result.winfo_children():
        widget.destroy()

    exactitud_label = tk.Label(frame_result, text=f"Exactitud de validación cruzada: {exactitud:.2f}%", font=("Arial", 14))
    exactitud_label.pack(pady=10)

    fig, ax = plt.subplots()
    cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicho')
    plt.ylabel('Actual')
    fig.colorbar(cax)

    ax.set_xticks(range(len(metrics)))
    ax.set_yticks(range(len(metrics)))
    ax.set_xticklabels([str(i) for i in range(len(metrics))])
    ax.set_yticklabels([str(i) for i in range(len(metrics))])

    for i in range(len(metrics)):
        for j in range(len(metrics)):
            ax.text(j, i, str(conf_matrix[i, j]), va='center', ha='center', color='black')

    canvas = FigureCanvasTkAgg(fig, master=frame_result)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    result_text = "Métricas por clase:\n"
    for cls, values in metrics.items():
        result_text += f"Clase {cls} - Precisión: {values['precision']:.2f}, Recall: {values['recall']:.2f}, F1 Score: {values['f1_score']:.2f}\n"

    result_label = tk.Label(frame_result, text=result_text, justify="left", font=("Arial", 12))
    result_label.pack()

def guardar_dataset_procesado(datos, etiquetas, nombre_archivo):
    df = pd.DataFrame(datos)
    df['Etiqueta'] = etiquetas
    df.to_csv(nombre_archivo, index=False)
    print(f"Dataset procesado guardado como {nombre_archivo}")

def smote_oversampling(datos, k):
    clases = [dato[0] for dato in datos]
    clases_unicas, conteos = np.unique(clases, return_counts=True)
    clase_mayoritaria = clases_unicas[np.argmax(conteos)]
    clases_minoritarias = [clase for clase in clases_unicas if clase != clase_mayoritaria]

    nuevos_datos = datos.copy()
    for clase in clases_minoritarias:
        datos_minoritarios = [dato for dato in datos if dato[0] == clase]
        n_samples = len([dato for dato in datos if dato[0] == clase_mayoritaria]) - len(datos_minoritarios)
        synthetic_samples = []

        while len(synthetic_samples) < n_samples:
            i = random.randint(0, len(datos_minoritarios) - 1)
            nn = vecinos(datos, datos_minoritarios[i], k)
            nn = random.choice(nn)
            diff = [nn[j] - datos_minoritarios[i][j] for j in range(1, len(datos_minoritarios[i]))]
            gap = random.random()
            synthetic_sample = [datos_minoritarios[i][0]] + [datos_minoritarios[i][j] + gap * diff[j - 1] for j in range(1, len(datos_minoritarios[i]))]
            synthetic_samples.append(synthetic_sample)

        nuevos_datos.extend(synthetic_samples)

    return nuevos_datos

def ejecutar_algoritmo_synthetic():
    global datos_entrenamiento, encabezado
    if datos_entrenamiento:
        k = int(k_var.get())
        print("Aplicando SMOTE...")
        progress_var.set(0)
        root.update_idletasks()

        datos_balanceados = smote_oversampling(datos_entrenamiento, k)

        progress_var.set(100)
        print("Oversampling completado. Nuevo conjunto de datos balanceado guardado como 'dataset_procesado.csv'.")
        guardar_dataset_procesado(datos_balanceados, [dato[0] for dato in datos_balanceados], 'dataset_procesado.csv')

        mostrar_grafica_barras(datos_entrenamiento, "Original Dataset", frame_barras)
        mostrar_grafica_barras(datos_balanceados, "Dataset Procesado", frame_result)

        # Validación cruzada con el dataset balanceado
        X = np.array([x[1:] for x in datos_balanceados])
        y = np.array([x[0] for x in datos_balanceados])
        knn = KNeighborsClassifier(n_neighbors=k)
        kf = KFold(n_splits=10)
        y_pred = cross_val_predict(knn, X, y, cv=kf)
        cross_val_acc = accuracy_score(y, y_pred)
        print('Exactitud de validación cruzada:', cross_val_acc * 100, '%')

        conf_matrix = confusion_matrix(y, y_pred)
        metrics = precision_recall_fscore_support(y, y_pred, average=None)
        metrics_dict = {label: {'precision': precision, 'recall': recall, 'f1_score': f1} 
                        for label, precision, recall, f1 in zip(np.unique(y), metrics[0], metrics[1], metrics[2])}
        mostrar_resultados(cross_val_acc * 100, conf_matrix, metrics_dict)

def ejecutar_algoritmo():
    algoritmo = algoritmo_var.get()
    if algoritmo == "KNN":
        main_knn()
    elif algoritmo == "Synthetic Sampling + KNN":
        ejecutar_algoritmo_synthetic()
    else:
        messagebox.showerror("Error", "Algoritmo no soportado")

# Configuración de la interfaz gráfica
root = tk.Tk()
root.title("Clasificación con KNN y Oversampling Sintético")
root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}")

frame_top = tk.Frame(root)
frame_top.pack(side="top", fill="x")

frame_progress = tk.Frame(root)
frame_progress.pack(side="top", fill="x")

frame_result = tk.Frame(root, height=300)
frame_result.pack(side="bottom", fill="both", expand=True)

frame_dataset = tk.Frame(root)
frame_dataset.pack(side="top", fill="both", expand=True)

frame_barras = tk.Frame(root)
frame_barras.pack(side="top", fill="both", expand=True)

# Variables
algoritmo_var = tk.StringVar(value="KNN")
k_var = tk.StringVar(value="1")
progress_var = tk.DoubleVar()

# Componentes de la interfaz
tk.Button(frame_top, text="Cargar Archivo CSV", command=cargar_archivo).pack(side="left", padx=10)
tk.Label(frame_top, text="Algoritmo:").pack(side="left")
tk.OptionMenu(frame_top, algoritmo_var, "KNN", "Synthetic Sampling + KNN").pack(side="left", padx=10)
tk.Label(frame_top, text="K:").pack(side="left")
tk.Entry(frame_top, textvariable=k_var, width=5).pack(side="left", padx=10)
tk.Button(frame_top, text="Ejecutar Algoritmo", command=ejecutar_algoritmo).pack(side="left", padx=10)

# Barra de progreso
progress_label = tk.Label(frame_progress, text="Progreso:")
progress_label.pack(side="left", padx=10)
progress_bar = ttk.Progressbar(frame_progress, variable=progress_var, maximum=100)
progress_bar.pack(fill="x", expand=True)

# Iniciar el bucle de la interfaz gráfica
root.mainloop()
