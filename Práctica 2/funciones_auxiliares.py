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
    n_columnas = 2

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


# Función que genera una representanción en pirámide Gaussiana
# Entrada:
#  - imagen: nombre del archivo al cual se le va a aplicar el filtro
#  - levels: niveles de la pirámide
#  - borde: borde que queremos aplicar a la imagen
# Salida: visualización de una imagen en pirámide Gaussiana
def piramide_gaussiana(imagen, levels, borde):

    # Genero una copia de la imagem de entrada
    gaussianPyramid = [imagen.copy()]

    # Recorro los niveles
    for i in range(1, levels):
        # Obtengo cada imagen con su tamaño y su alisado
        img = cv2.pyrDown(gaussianPyramid[i - 1], borderType=borde)
        # Y la meto en una lista
        gaussianPyramid.append(img)

    # Compruebo si la imagen es en color
    if (len(np.shape(imagen))) == 3:

        # Creo una imagen, con el tamaño de todas las imágenes a meter
        nueva_imagen = np.empty((np.shape(imagen)[0], math.ceil(np.shape(imagen)[1] * 1.5),
                                 np.shape(imagen)[2]))

        # Meto la primera imagen a mano (estará a la izquierda)
        nueva_imagen[0:np.shape(gaussianPyramid[0])[0],
        0:np.shape(gaussianPyramid[0])[1], 0:3] = gaussianPyramid[0]
        # Meto la segunda imagen a mano (estará a la derecha)
        nueva_imagen[0:np.shape(gaussianPyramid[1])[0],
        np.shape(gaussianPyramid[0])[1]:math.ceil(np.shape(gaussianPyramid[1])[1] + np.shape(gaussianPyramid[0])[1]),
        0:3] = gaussianPyramid[1]

        # Recorro los niveles que me quedan
        for i in range(2, levels):

            # Calculo la altura de cada imagen, para añadirla a continuación de la segunda imagen
            # Por eso voy sumando las alturas, para que así vaya una detrás de otra
            altura = 0
            for j in range(1, i):
                altura = altura + np.shape(gaussianPyramid[j])[0]

            # Ya tengo mi imagen con la pirámide en color (los voy metiendo en mi imagen ventana)
            nueva_imagen[altura:(np.shape(gaussianPyramid[i])[0] + altura),
            np.shape(gaussianPyramid[0])[1]:math.ceil(
                np.shape(gaussianPyramid[i])[1] + np.shape(gaussianPyramid[0])[1]),
            0:3] = gaussianPyramid[i]

    # Compruebo si la imagen es en blanco y negro
    else:

        # Creo una imagen, con el tamaño de todas las imágenes a meter
        nueva_imagen = np.empty((np.shape(imagen)[0], math.ceil(np.shape(imagen)[1] * 1.5)))

        # Meto la primera imagen a mano (estará a la izquierda)
        nueva_imagen[0:np.shape(gaussianPyramid[0])[0],
        0:np.shape(gaussianPyramid[0])[1]] = gaussianPyramid[0]
        # Meto la segunda imagen a mano (estará a la derecha)
        nueva_imagen[0:np.shape(gaussianPyramid[1])[0],
        np.shape(gaussianPyramid[0])[1]:math.ceil(np.shape(gaussianPyramid[1])[1] + np.shape(gaussianPyramid[0])[1])] = \
        gaussianPyramid[1]

        # Recorro los niveles que me quedan
        for i in range(2, levels):

            # Calculo la altura de cada imagen, para añadirla a continuación de la segunda imagen
            # Por eso voy sumando las alturas, para que así vaya una detrás de otra
            altura = 0
            for j in range(1, i):
                altura = altura + np.shape(gaussianPyramid[j])[0]

            # Ya tengo mi imagen con la pirámide en color (los voy metiendo en mi imagen ventana)
            nueva_imagen[altura:(np.shape(gaussianPyramid[i])[0] + altura),
            np.shape(gaussianPyramid[0])[1]:math.ceil(
                np.shape(gaussianPyramid[i])[1] + np.shape(gaussianPyramid[0])[1])] = gaussianPyramid[i]

    # Devuelvo la pirámide
    # return nueva_imagen.astype(np.uint8)
    return gaussianPyramid, nueva_imagen.astype(np.uint8)


# Función que desenfoca una imagen usando un filtro gaussiana
# Entrada:
#  - imagen: nombre del archivo al cual se le va a aplicar el filtro
#  - sigma: valor del sigma
#  - borde: borde que queremos aplicar a la imagen
# Salida: devuelve la imagen con el filtro aplicado
def alisar_imagen(imagen, sigma, borde):

    # Calculamos el tamanio de la mascara
    tamanio = 3 * sigma * 2 + 1

    # Aplicamos el filtro a la imagen
    blur = cv2.GaussianBlur(imagen, ksize=(tamanio, tamanio),
                            sigmaX=sigma, sigmaY=sigma, borderType=borde)

    # Cambiamos la imagen a una profundidad(color) de 8 o 16 bits sin signo int (8U, 16U)
    # o 32 bit flotante (32F)
    return blur.astype(np.uint8)


# Función que aplica convolucion primero por filas y luego por columnas
# Entrada:
#  - imagen: nombre del archivo al cual se le va a aplicar el filtro
#  - borde: borde que queremos aplicar a la imagen
#  - mask_fila: mascara que aplicaremos por filas
#  - mask_col: mascara que aplicaremos por columnas
# Salida: devuelve la imagen convolucionada
def convolucion_separable(imagen, borde, mask_fila=None, mask_col=None):

    # Copio la imagen de entrada
    nueva_imagen = np.copy(imagen)

    # Si imagen en escala de grises B/N
    if len(np.shape(imagen)) == 2:

        if mask_fila is not None:

            # Iteramos por filas
            for i in range(np.shape(imagen)[0]):
                # Aplicamos convolución
                fila = cv2.filter2D(imagen[i, :], -1, mask_fila, borderType=borde)

                # debemos poner fila[:,0] ya que filter2D crea una estructura de canales
                # pero la imagen no tiene canales
                nueva_imagen[i, :] = fila[:, 0]

        if mask_col is not None:

            nueva_imagen = nueva_imagen.transpose(1, 0)  # Transponemos para cambiar filas por columnas

            # Iteramos ahora por columnas (igual que con las filas)
            for i in range(np.shape(imagen)[1]):
                # Aplicamos convolución
                columna = cv2.filter2D(nueva_imagen[i, :], -1, mask_col, borderType=borde)
                nueva_imagen[i, :] = columna[:, 0]

            # Volvemos a trasponer para obtener la imagen original alisada
            nueva_imagen = nueva_imagen.transpose(1, 0)

    else:  # Imagen en color

        if mask_fila is not None:

            # Iteramos filas
            for i in range(np.shape(imagen)[0]):
                # Aplicamos convolución por cada canal
                fila = cv2.filter2D(imagen[i, :, :], -1, mask_fila, borderType=borde)
                nueva_imagen[i, :, :] = fila

        if mask_col is not None:

            nueva_imagen = nueva_imagen.transpose(1, 0, 2)  # Transponemos para cambiar filas por columnas

            # Iteramos ahora por columnas
            for i in range(np.shape(imagen)[1]):
                # Aplicamos convolución por cada canal
                columna = cv2.filter2D(nueva_imagen[i, :, :], -1, mask_col, borderType=borde)
                nueva_imagen[i, :, :] = columna

            # Volvemos a trasponer para obtener la imagen original alisada
            nueva_imagen = nueva_imagen.transpose(1, 0, 2)

    return nueva_imagen.astype(np.uint8)


# Función de convolución con núcleo de primera derivada
# Entrada:
#  - imagen: nombre del archivo al cual se le va a aplicar el filtro
#  - sigma: valor del sigma
#  - borde: borde que queremos aplicar a la imagen
# Salida: visualización de imágenes aplicando la primera derivada
def primera_derivada(imagen, sigma, borde):
    # Calculamos el tamanio
    tamanio = 3 * sigma * 2 + 1

    # Hacemos la primera derivada del núcleo y nos devolverá una para X y otra para Y
    mascara = cv2.getDerivKernels(dx=1, dy=1, ksize=tamanio)
    # Le asignamos la primera derivada a las filas (X)
    mascara_fila = mascara[0]

    # Usamos la convolución del apartado C
    # Para conseguir los bordes horizontales --> por eso necesitas pasar por columnas
    bordes_horizontales = convolucion_separable(imagen, cv2.BORDER_REFLECT, None, mascara_fila)

    # Le asignamos la primera derivada a las filas (Y)
    mascara_col = mascara[1]
    # Para conseguir los bordes verticales --> por eso necesitas pasar por filas (es cuando cruza)
    bordes_verticales = convolucion_separable(imagen, cv2.BORDER_REFLECT, mascara_col, None)

    # Normalizamos imágenes de derivadas
    bordes_horizontales = cv2.normalize(bordes_horizontales, bordes_horizontales, alpha=0, beta=255,
                                        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    bordes_verticales = cv2.normalize(bordes_verticales, bordes_verticales, alpha=0, beta=255,
                                      norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # representar_imagenes([imagen, bordes_horizontales,bordes_verticales],
    #	['Original alisada','1ª Bordes Horizontales','1ª Bordes Verticales'])

    return bordes_horizontales, bordes_verticales
