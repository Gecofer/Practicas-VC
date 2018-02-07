#! /usr/bin/env python
# -*- coding: utf-8 -*-

### FUNCIONES AUXILIARES ###

# Importamos las librerías necesarias para desarrollar la práctica
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math


# Función que lee una imagen (leemos una imagen con la funcion de opencv 'imread' con el flag especificado)
# Entrada:
#  - filename: nombre del archivo desde el que se lee la imagen
#  - flag_color: variable booleana que nos indica si leemos una imagen en color o en escala de grises
# Salida: devuelve la matriz (el objeto) donde se almacena
def leer_imagen(filename, flag_color=True):
    if (flag_color):  # Si flag_color es True, carga una imagen en color de 3 canales
        image = cv2.imread(filename, flags=1)

    else:  # Si flag_color es False, carga una imagen en escala de grises
        image = cv2.imread(filename, flags=0)

    return image


# Función que representa imágenes con sus títulos en una misma ventana
# Entrada:
#  - lista_imagen_leida: lista de las imágenes a mostrar
#  - lista_titulos: lista de títulos de cada imagen a mostrar
# Salida: visualización de las imágenes con sus títulos en una misma ventana
def representar_imagenes(lista_imagen_leida, lista_titulos):

    # Comprobamos que el numero de imágenes corresponde con el número de títulos pasados
    if len(lista_imagen_leida) != len(lista_titulos):
        print("No hay el mismo número de imágenes que de títulos.")
        return -1  # No hay el mismo numero de imágenes que de títulos

    # Calculamos el numero de imagenes
    n_imagenes = len(lista_imagen_leida)

    # Establecemos por defecto el numero de columnas
    n_columnas = 5

    # Calculamos el número de filas
    n_filas = (n_imagenes // n_columnas) + (n_imagenes % n_columnas)

    # Establecemos por defecto un tamaño a las imágenes
    # plt.figure(figsize=(9,9))

    # Recorremos la lista de imágenes
    for i in range(0, n_imagenes):

        plt.subplot(n_filas, n_columnas, i+1) # plt.subplot empieza en 1

        # if (len(np.shape(lista_imagen_leida[i]))) == 2: # Si la imagen es en gris
        if (len(np.shape(lista_imagen_leida[i]))) == 2:  # Si la imagen es en gris
            plt.imshow(lista_imagen_leida[i], cmap='gray')
        else:  # Si la imagen es en color
            plt.imshow(cv2.cvtColor(lista_imagen_leida[i], cv2.COLOR_BGR2RGB))

        plt.title(lista_titulos[i])  # Añadimos el título a cada imagen

        plt.xticks([]), plt.yticks([])  # Para ocultar los valores de tick en los ejes X e Y

    plt.show()

    plt.waitforbuttonpress()
