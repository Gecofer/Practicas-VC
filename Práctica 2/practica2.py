#! /usr/bin/env python
# -*- coding: utf-8 -*-

#######################################################################################################################
# Trabajo 2: Detección de puntos relevantes y Construcción de panoramas                                               #
# Curso 2017/2018                                                                                                     #
# Gema Correa Fernández                                                                                               #
#######################################################################################################################

# Importamos las librerías necesarias para desarrollar la práctica
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
from funciones_auxiliares import *

# _#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#
#                                                       EJERCICIO 1                                                    #
# _#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#

# ------------------------------------------------------------------------------------------------------------------- #
#                                                       Apartado A                                                    #
#                                                                                                                     #
#           Extrae la lista potencial de puntos Harris a distintas escalas de una imagen de nivel de gris             #
# ------------------------------------------------------------------------------------------------------------------- #

'''
Función que obtiene los puntos harris de una imagen

Sobre cada nivel de la pirámide usar la función OpenCV cornerEigenValsAndVecs para extraer la información de 
autovalores y autovectores de la matriz Harris en cada píxel (fijar valores de blockSize y ksize equivalentes al uso 
de máscaras gaussianas de sigmaI=1.5 y sigmaD=1 respectivamente)

cornerEigenValsAndVecs devuelve 6 matrices (λ1, λ2, x1, y1, x2, y2) donde:
    - λ1, λ2 son los autovalores no ordenados de M
    - x1, y1 son los autovectores de λ1
    - x2, y2 son los autovectores de λ2
        
Usar uno de los criterios de selección estudiados a partir de los autovalores y crear una matriz con el valor del 
criterio selección asociado a cada píxel (para el criterio Harris usar k=0.04) (criterio en la pagina 36 del T2)
'''
def obtenerPuntosHarris(lista_images, block_size, k_size, umbral):

    '''
    :param lista_images: vector con las imágenes de la pirámide gaussiana
    :param block_size: variable entera que indica tamaño del vecindario
    :param k_size: sigma de la derivada
    :param umbral: variable basada en un criterio para la aceptacion de puntos Harris

    :return: la imagen binaria representando los puntos de Harris y su correspondiente matriz
    '''

    # Variable donde se guarda el resultado de aplicar el criterio de selección por cada pixel
    matriz_criterio_seleccion = []

    # Por cada nivel de la piramide realizamos la deteccion de puntos Harris
    for nivel in lista_images:

        # Obtenemos los autovalores y autovectores propios asociados a la matriz Harris en cada pixel
        informacion_pixel = cv2.cornerEigenValsAndVecs(src=nivel, blockSize=block_size, ksize=k_size)

        # Separamos la informacion obtenida
        lambd = cv2.split(informacion_pixel)

        # Calculamos los valores del criterio de seleccion Harris con k=0.04
        # determinant(H) - k * (trace(H))^2 --> donde det(H) = λ1 * λ2, k = 0.04, trace(H) = λ1 + λ2
        k = 0.04
        matriz_criterio_seleccion.append(lambd[0]*lambd[1] - k * ((lambd[0]+lambd[1])*(lambd[0]+lambd[1])))

    # Creamos la matriz binaria de los puntos de Harris, los cuales tendrán valor 255
    imagen_binaria = [((nivel >= umbral)*255) for nivel in matriz_criterio_seleccion]

    # Devolvemos la matriz con el valor del criterio selección asociado a cada píxel y su respectiva imagen binaria
    return matriz_criterio_seleccion, imagen_binaria


'''
Función que implementa la fase de supresión de valores no máximos sobre la matriz de puntos harris

Elimina como candidatos los píxeles que teniendo un valor alto de criterio de Harris no son máximos locales de su 
entorno para un tamaño de entorno fijo. Solo estamos interesados en píxeles cuyo valor Harris sea un máximo local.
'''
def supresionNoMaximos(puntos_harris, imagen_binaria, tam_ventana):

    '''
    :param puntos_harris: matrices obtenidas de los puntos harris
    :param imagen_binaria: imagen binaria obtenida de los puntos harris
    :param tam_ventana: variable entera que indica tamaño del vecindario

    :return: la imagen binaria de puntos Harris con la supresion de no maximos
    '''

    '''
    Función que nos dice si el valor del centro de una ventana es máximo local (la dimensión de la ventana es impar)
    '''
    def maximoLocalCentro (tam_ventana):

        # Obtenemos el número de filas y de columnas de la ventana
        centro_filas = np.shape(tam_ventana)[0]
        centro_col = np.shape(tam_ventana)[1]

        # Calculamos el valor máximo de la ventana
        maximo = np.argmax(tam_ventana)

        # Calculamos el valor del centro de nuestra ventana
        centro = (centro_filas * centro_col)//2

        # Si es maximo local devolvemos True, sino devolvemos False
        if (maximo == centro): return True
        else: return False

    '''
    Función que sobre una imagen binaria inicializada a 255, sea capaz de modificar a 0 todos los pixeles de un
    rectangulo dado, es decir, pone en negro todos los píxeles de un entorno menos el central
    '''
    def modificarPixelesEntorno(imagen_binaria, rectangulo, i , j):

        # Calculamos el centro de la ventana
        centro_filas = rectangulo/2
        centro_col = rectangulo/2

        # Si no estoy en el centro de la ventana, modificar a 0 todos los pixeles de un rectángulo dado
        if not (i == centro_filas) and not (j == centro_col):
            imagen_binaria[rectangulo,rectangulo] = 0

    '''
    Fijar un tamaño de entorno/ventana y recorrer la matriz binaria ya creada preguntando, en cada posición de valor 
    255, si el valor del pixel correspondiente de la matriz Harris es máximo local o no
    '''
    for nivel in range(6):

        # Recorrer la matriz binaria ya creada, preguntando en cada posición por el valor 255
        harris = (imagen_binaria[nivel])
        harris_255 = (np.where(harris == 255))

        # Obtenemos el tamaño de la lista con las posiciones a 255
        len_harris_255 = len(harris_255[1])

        # Comprobamos si el valor del pixel correspondiente de la matriz Harris es máximo local o no
        for i in range(len_harris_255):

            # Calculamos el valor de las filas y las columnas
            fila = harris_255[0][i]
            columna = harris_255[1][i]

            # Calculamos las cuatro esquinas del rectangulo
            izquierda = fila - tam_ventana
            derecha = fila + tam_ventana + 1
            abajo = columna - tam_ventana
            arriba = columna + tam_ventana + 1

            # Si es maximo local ponemos a negro a todos los píxeles de su entorno
            if fila >= tam_ventana and columna >= tam_ventana and maximoLocalCentro(puntos_harris[nivel][izquierda:derecha, abajo:arriba]):
                modificarPixelesEntorno(imagen_binaria[nivel], tam_ventana, fila, columna)

            # Si no es máximo local lo ponemos a 0
            else: imagen_binaria[nivel][fila, columna] = 0

    return imagen_binaria


'''
Ordenar de mayor a menor los puntos resultantes de acuerdo a su valor y seleccionar al menos los 500 puntos 
de mayor valor.
'''
def seleccionar500MejoresPuntosHarris(puntos_harris, imagen_binaria, tam_ventana):

    '''
    :param imagen: imagen donde visualizaremos los puntos
    :param puntos_harris: matrices obtenidas de los puntos harris
    :param imagen_binaria: imagen binaria obtenida de los puntos harris
    :param tam_ventana: variable entera que indica tamaño del vecindario

    :return: imagen con los mejores puntos harris
    '''

    # Acabamos de quedarnos con los puntos que no se han ido en la fase de supresión de valores no-máximos
    # Los ordenamos por su valor de Harris para quedarnos con al menos los 500 mejores

    # Nos creamos las variables necesarias para realizar los cálculos
    indices_harris_255 = []     # lista donde gaurdamos los índices de los puntos harris (a 255)
    puntos_harris_255 = []      # lista donde guardamos los puntos harris (a 255)
    mejores_puntos_harris = []  # lista que guarda los mejores puntos harris

    for nivel in range(6):

        # Nos quedamos con los índices que corresponden con puntos de harris, es decir a 255
        harris = (imagen_binaria[nivel])
        indices_harris_255 = (np.where(harris == 255))

        # Nos quedamos con el valor de harris, una vez obtenido el índice
        puntos_harris_255 = puntos_harris[nivel][indices_harris_255]

        # Cogemos las coordenadas x,y de los puntos y las juntamos
        x = indices_harris_255[0]
        y = indices_harris_255[1]

        # Los metemos en lista (x,y, nivel, valor)
        nueva = [(x[i],y[i], nivel, puntos_harris_255[i]) for i in range(len(indices_harris_255[0]))]

        # Vamos metiendo los valores en una lista
        mejores_puntos_harris = mejores_puntos_harris + nueva

    # Y de esa lista los ordenamos de mayor a menor y nos quedamos con los 500 primeros
    valor = sorted(mejores_puntos_harris[0:500], key = lambda tup:tup[3])[::-1]

    return valor


'''
Función dibuja un círculo y su orientación en cada punto (en la imagen original)
'''
def dibujarCirculoLineas(imagen, puntos_harris, titulo, orientaciones = None, usarOrientaciones = False):

    '''
    Mostrar el resultado dibujando sobre la imagen original un círculo centrado en cada punto y de radio
    proporcional al valor del sigma usado para su detección (ver circle()).
    '''

    # Convertimos a array las orientaciones
    array = np.array(orientaciones)

    # Recorremos para todos los puntos harris
    for punto in puntos_harris:

        # Obtenemos la escala
        escala = punto[2]

        # Si hay mas de una escala en la imagen, el círculo deberá estar centrado en cada punto y de radio proporcional a la escala
        if escala > 0:

            # Dibujamos los círculos en la imagen original
            cv2.circle(img=imagen, center=(int(punto[1]*escala*2), int(punto[0]*escala*2)), radius=(escala+1)*6, color=0, thickness=0)

            # Si usamos los ángulos
            if usarOrientaciones:

                # Comparamos las tuplas de puntos refinados con las orientaciones
                # Si (x,y,escala) es igual entre ambas, nos quedamos con ese valor
                p = array[(array[:,0]==punto[0]) & (array[:,1]==punto[1]) & (array[:,2]==punto[2])]

                # Obtenemos el ángulo de ese punto
                angulos = p[:,3]

                cv2.arrowedLine(img=imagen, pt1=(int(punto[1])*(escala+1), int(punto[0])*(escala+1)),
                                pt2=(int(punto[1])*(escala+1) + math.floor(np.sin(angulos)*(escala+1)*6),
                                     int(punto[0])*(escala+1) + math.floor(np.cos(angulos)*(escala+1)*6)),
                                color=0,thickness=0)

        # Si solo tenemos puntos de la imagen original
        else:

            # Dibujamos los círculos en la imagen original
            cv2.circle(img=imagen, center=(punto[1], punto[0]), radius=8, color=0, thickness=0)

            # Si usamos los ángulos
            if usarOrientaciones:

                # Comparamos (x,y,escala) de ambas tuplas y nos quedamos con ese punto
                p = array[(array[:,0]==punto[0]) & (array[:,1]==punto[1]) & (array[:,2]==punto[2])]

                # Obtenemos el ángulo de ese punto
                angulos = p[:,3]

                cv2.line(img=imagen, pt1=(int(punto[1]), int(punto[0])),
                                pt2=(int(punto[1])+ math.floor(np.sin(angulos)*8),
                                     int(punto[0]) + math.floor(np.cos(angulos)*8)),
                                color=0,thickness=0)

    #cv2.imwrite("imagenes/harris.jpg", imagen)
    representar_imagenes([imagen], [titulo])


# ------------------------------------------------------------------------------------------------------------------- #
#                                                       Apartado B                                                    #
#                                                                                                                     #
#           Extraer los valores (cx, cy, escala) de cada uno de los puntos y refinar su posición                      #
# ------------------------------------------------------------------------------------------------------------------- #

# Función para refinar los puntos sacados en el apartado A
def refinarPuntosHarris(lista_imagenes, puntos_harris):

    '''
    :param lista_imagenes: la pirámide gaussiana (vector con cada escala)
    :param selected_points: puntos harris seleccionados

    :return: los puntos refinados a nivel de subpixel
    '''

    # Nos definimos las variables necesarias
    lista = []
    puntos_refinados = []

    # Seleccionamos las coordenadas X e Y de nuestros puntos harris
    selected_points_nueva = [(i[0],i[1]) for i in puntos_harris]

    # Convertimos en Array
    puntos = np.array(puntos_harris)

    # Recorremos todas las escalas que haya
    for i in range(int(np.max((puntos[:,2])))+1):

        # Ordenar por escala, y hacer subgrupos
        selected_points_nueva = [[punto[0],punto[1]] for punto in puntos[puntos[:,2]==i]]

        # Convertimos a array esa lista
        lista = np.array(selected_points_nueva, dtype=np.float32).copy()

        # Refinamos a nivel de subpixel
        cv2.cornerSubPix(image=lista_imagenes[i], corners=lista, winSize=(3, 3), zeroZone=(-1, -1),
                         criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 40, 0.001))

        # Actualizamos la tupla con las nuevas coordenadas de X e Y
        for punto in lista:
            puntos_refinados.append((punto[0], punto[1],i))

    return puntos_refinados


# ------------------------------------------------------------------------------------------------------------------- #
#                                                       Apartado C                                                    #
#                                                                                                                     #
#           Calcular la orientacion relevante de cada punto Harris usando el arco tangente de cada punto              #
# ------------------------------------------------------------------------------------------------------------------- #

# Función que obtiene las orientaciones relevantes de cada punto usando el arco tange del gradiente en cada punto
def obtenerOrientacionPunto(lista_imagenes, puntos_refinados):

    # Tupla donde guardaremos las orientaciones (x, y, escala, orientacion)
    orientaciones = []

    # Convertimos a array los puntos
    puntos = np.array(puntos_refinados)

    # Recorremos para cada escala de la imagen
    for i in range(int(np.max((puntos[:,2])))+1):

        # Calculamos las derivadas en X e Y de la imagen y las alisamos con sigma=4.5
        img1 = alisar_imagen(lista_imagenes[i], sigma=5, borde=cv2.BORDER_DEFAULT)
        derivada_x, derivada_y = primera_derivada(img1, 1, cv2.BORDER_DEFAULT)

        # Me quedo con los puntso refinados de una la escala i
        refined_points_nueva = [[punto[0],punto[1]] for punto in puntos[puntos[:,2]==i]]

        # Para cada punto añado (x,y,escala,orientacion)
        for punto in refined_points_nueva:
             orientaciones.append((punto[0],punto[1],i,((np.arctan2(derivada_y[int(punto[0]),int(punto[1])],
                                                                    derivada_x[int(punto[0]),int(punto[1])]))*180/np.pi)))

    # Convierto a array
    o = np.array(orientaciones)

    return orientaciones


# ------------------------------------------------------------------------------------------------------------------- #
#                                                       Apartado D                                                    #
# ------------------------------------------------------------------------------------------------------------------- #

# Función que convierte mis puntos Harris en un vector de Keypoints
def keypoints(puntos_harris, imagen):

    keypoints = []

    # Recorremos nuestros puntos Harris
    for punto in puntos_harris:
        if punto[2] == 0:
            keypoints.append(cv2.KeyPoint(x=punto[1], y=punto[0], _size=(punto[2]+1)*6))

    # Usamos el descriptor de SIFT
    sift = cv2.xfeatures2d.SIFT_create()

    # Obtenemos los keypoints para dibujarlos
    keypoints, descriptor = sift.compute(imagen, keypoints)

    # Dibujamos los puntos
    resultado = cv2.drawKeypoints(imagen, keypoints, imagen.copy(), color=(0,0,255), flags=0)

    # Los representamos
    representar_imagenes([resultado], ["Mis Keypoints"])

    return (keypoints, descriptor)



# _#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#
#                                                       EJERCICIO 2                                                    #
# _#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#

'''
Usar el detector descriptor SIFT de OpenCV sobre las imágenes de Yosemite.rar (cv2.xfeatures2d.SIFT_create()). 
Extraer sus listas de keyPoints y descriptores asociados. Establecer las correspondencias existentes entre ellos 
usando el objeto BFMatcher de OpenCV. Valorar la calidad de los resultados obtenidos en términos de correspondencias 
válidas usando los criterios de correspondencias "BruteForce+crossCheck" y "Lowe-Average-2NN"
'''

# Función para criterios de correspondencia "BruteForce+crossCheck"
def establecerCorrespondenciasFuerzaBruta(imagen1, imagen2):

    '''
    :param imagen1: queryImage
    :param imagen2: trainImage
    '''

    # Usamos el detector SIFT
    sift = cv2.xfeatures2d.SIFT_create()

    # Extraemos sus listas de keyPoints y descriptores asociados
    keypoints_imagen1, descriptors_imagen1 = sift.detectAndCompute(image=imagen1, mask=None)
    keypoints_imagen2, descriptors_imagen2 = sift.detectAndCompute(image=imagen2, mask=None)

    # Sacamos las correspondencias por fuerza bruta usando un objeto de tipo BFMatcher
	# le pasamos que use la norma NORM_L2 y cross check a TRUE
    correspondecias = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)

    # Sacamos las correspondencias
    matches = correspondecias.match(descriptors_imagen1, descriptors_imagen2)

    # Ordenamos las correspondencias por mejor distancia
    matches = sorted(matches, key=lambda x: x.distance)

    # Cogemos 100 de las correspondecias extraidas para apreciar la calidad de los resultados
    numero_correspondencias = 100

    resultado = cv2.drawMatches(img1=imagen1, keypoints1=keypoints_imagen1, img2=imagen2, keypoints2=keypoints_imagen2,
                                matches1to2=matches[:numero_correspondencias], outImg=imagen2.copy(), flags=0)

    #cv2.imwrite("imagenes/correspondenciaFB.jpg", resultado)
    representar_imagenes([resultado],['Correspondencias por Fuerza Bruta'])


# Función para criterios de correspondencia "Lowe-Average-2NN"
def establecerCorrespondencias2NN(imagen1, imagen2):

    '''
    :param imagen1: queryImage
    :param imagen2: trainImage
    '''

    # Usamos el detector SIFT (Inicializar)
    sift = cv2.xfeatures2d.SIFT_create()

    # Extraemos sus listas de keyPoints y descriptores asociados
    keypoints_imagen1, descriptors_imagen1 = sift.detectAndCompute(image=imagen1, mask=None)
    keypoints_imagen2, descriptors_imagen2 = sift.detectAndCompute(image=imagen2, mask=None)

    # Sacamos las correspondencias por fuerza bruta usando un objeto de tipo BFMatcher
	# le pasamos que use la norma NORM_L2 y cross check a FALSE
    correspondecias = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)

    # Usaremos BFMatcher.knnMatch() para obtener los k mejores coincidencias, tomamos k=2
    matches = correspondecias.knnMatch(descriptors_imagen1, descriptors_imagen2, k=2)

    # Aplicamos ratio test
    aceptadas = []
    for m, n in matches:
        if m.distance < 0.70 * n.distance:
            aceptadas.append([m])

    # Cogemos 100 de las correspondecias extraidas para apreciar la calidad de los resultados
    numero_correspondencias = 100

    resultado = cv2.drawMatchesKnn(img1=imagen1, keypoints1=keypoints_imagen1, img2=imagen2, keypoints2=keypoints_imagen2,
                                matches1to2=aceptadas[:numero_correspondencias], outImg=imagen2.copy(), flags=2)

    #cv2.imwrite("imagenes/correspondencia2NN.jpg", resultado)
    representar_imagenes([resultado],['Correspondencias 2NN'])


# Establecer correspondencias con los keypoints obtenidos en el apartado anterior
def correspondenciasMisKeyPoints (keypoints1, imagen1, descriptor1, keypoints2, imagen2, descriptor2):

    # Sacamos las correspondencias por fuerza bruta usando un objeto de tipo BFMatcher
	# le pasamos que use la norma NORM_L2 y cross check a TRUE
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Sacamos las correspondencias
    matches = bf.match(descriptor1, descriptor2)

    # Ordenamos las correspondencias por mejor distancia
    matches = sorted(matches, key=lambda x: x.distance)

    # Dibujo los puntos con sus correspondencias
    resultado = cv2.drawMatches(imagen1, keypoints1, imagen2, keypoints2, matches[:50], outImg=imagen2.copy(), flags=0, matchColor=(0,0,255))

    #cv2.imwrite("imagenes/prueba.png", resultado)
    representar_imagenes([resultado],["Correspondencias entre misKeypoints"])



# _#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#
#                                                       EJERCICIO 3                                                    #
# _#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#

'''
Escribir una función que genere un Mosaico de calidad a partir de N=3 imágenes relacionadas por homografías, sus listas 
de keyPoints calculados de acuerdo al punto anterior y las correspondencias encontradas entre dichas listas. 
Estimar las homografías entre ellas usando la función cv2.findHomography(p1,p2,CV_RANSAC,1). 

Para el mosaico 
será necesario. a) definir una imagen en la que pintaremos el mosaico; b) definir la homografía que lleva cada una de 
las imágenes a la imagen del mosaico; c) usar la función cv2. warpPerspective() para trasladar cada imagen al mosaico 
(ayuda: mirar el flag BORDER_TRANSPARENT de warpPerspective para comenzar). (2.5 puntos)
'''

# Función que genera un Mosaico para N=3 imágenes adyacentes
def mosaicoN3(imageA, imageB, imageC):

    # Usamos el detector SIFT (Inicializar)
    descriptor = cv2.xfeatures2d.SIFT_create()

    # Extraemos las listas de keyPoints y descriptores asociados para cada imagen
    keypointsA, descriptorA = descriptor.detectAndCompute(imageA, None)
    keypointsB, descriptorB = descriptor.detectAndCompute(imageB, None)
    keypointsC, descriptorC = descriptor.detectAndCompute(imageC, None)

    # Convertimos los objetos de Keypoint a Arrays
    kpsA = np.float32([kp.pt for kp in keypointsA])
    kpsB = np.float32([kp.pt for kp in keypointsB])
    kpsC = np.float32([kp.pt for kp in keypointsC])

    # Primero calculamos las correspondencias entre la imagen A y la imagen B
    # -----------------------------------------------------------------------

    # Sacamos las correspondencias por fuerza bruta usando un objeto de tipo BFMatcher
	# le pasamos que use la norma NORM_L2 y cross check a FALSE
    matcher = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)

    # Usaremos BFMatcher.knnMatch() para obtener los k mejores coincidencias, tomamos k=2
    best_matches = matcher.knnMatch(descriptorA, descriptorB, k=2)

    # Aplicamos ratio test para obtener la mejor distancia
    matches = []
    for m in best_matches:
        if len(m) == 2 and m[0].distance < m[1].distance:
            matches.append((m[0].trainIdx, m[0].queryIdx))

    # Creamos la homografia si existen un mínimo de correspondencias entre las dos imágenes
    if len(matches) > 4:
        # construct the two sets of points
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])

        # compute the homography between the two sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 1)


    # PARA IMAGEN B E IMAGEN C
    # Primero calculamos las correspondencias entre la IMAGEN B y la IMAGEN C
    # -----------------------------------------------------------------------

    # Sacamos las correspondencias por fuerza bruta usando un objeto de tipo BFMatcher
	# le pasamos que use la norma NORM_L2 y cross check a FALSE
    matcher2 = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)

    # Usaremos BFMatcher.knnMatch() para obtener los k mejores coincidencias, tomamos k=2
    best_matches2 = matcher2.knnMatch(descriptorB, descriptorC, k=2)

    # Aplicamos ratio test para obtener la mejor distancia
    matches2 = []
    for m in best_matches2:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if len(m) == 2 and m[0].distance < m[1].distance:
            matches2.append((m[0].trainIdx, m[0].queryIdx))

    # computing a homography requires at least 4 matches
    # minimo de puntos en comun
    if len(matches2) > 4:
        # construct the two sets of points
        ptsC = np.float32([kpsB[i] for (_, i) in matches2])
        ptsD = np.float32([kpsC[i] for (i, _) in matches2])

        # compute the homography between the two sets of points
        (H2, status2) = cv2.findHomography(ptsC, ptsD, cv2.RANSAC, 1)

    # Primero tenemos que transladar la imagen central a su homografia al mosaico
    translacion = np.matrix([[1, 0, imageB.shape[0]], [0, 1, imageB.shape[1]], [0, 0, 1]], dtype=np.float32)

    # La pegamos con warpPerspective
    result = cv2.warpPerspective(imageB, translacion, dsize=(imageB.shape[1]*4, imageB.shape[0]*4),  borderMode=cv2.BORDER_TRANSPARENT)

    # Calcula la homografia respecto del mosaico
    cv2.warpPerspective(imageA, translacion*H, dst=result, borderMode=cv2.BORDER_TRANSPARENT,
                                  dsize=(imageA.shape[1]*4, imageA.shape[0]*4))

    cv2.warpPerspective(imageC, translacion*H*H2, dst=result, borderMode=cv2.BORDER_TRANSPARENT,
                                 dsize=(imageC.shape[1]*4, imageC.shape[0]* 4))

    return result


# _#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#
#                                                       EJERCICIO 4                                                    #
# _#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#

# Función que genera un Mosaico para N imágenes relacionadas
def mosaicoN(lista_imagenes):

    '''
    :param lista_imagenes: imágenes ordenadas de izquierda a derecha para generar el mosaico

    :return: mosaico de salida imprimido en un imagen con fondo negro.
    '''

    # Nos creamos las variables necesarias a usar
    coincidencias = []
    keypoints = []
    homografia = []
    descriptors = []

    # Clase para la coincidencia de descripciones keypoints
    # (query descriptor index, train descriptor index, train image index, and distance between descriptors)
    coincidencias.append(cv2.DMatch)

    # Para extraer los puntos clave y los descriptores utilizamos SIFT
    # Debemos tener acceso a la implementación original de SIFT, que están en el submódulo xfeatures2d
    sift = cv2.xfeatures2d.SIFT_create()

    # Para establecer las correspondencias entre las imágenes usaremos: Brute-force matcher create method
    correspondencias = cv2.BFMatcher( normType=cv2.NORM_L2, crossCheck=True)

    # Saco los keypoints y descriptores de cada imagen del mosaico
    for i in range(len(lista_imagenes)):

        # Detectar y extraer características de la imagen
        kps, desc = sift.detectAndCompute(image=lista_imagenes[i], mask=None)

        # Convertir los keypoints con estructura Keypoint a un Array
        kps = np.float32([kp.pt for kp in kps])

        # Guarda en un lista las características de cada imagen
        keypoints.append(kps)
        descriptors.append(desc)

    # Obtengo el vector de correspondencias y la homografía de cada par de imagenes adyacentes en el mosaico horizontal
    for i in range (len(lista_imagenes)-1):

        # Obtengo las correspondencias de la imagen i con la i+1
        matches = correspondencias.match(descriptors[i], descriptors[i+1])
        # Ordeno las coincicencias por el orden de la distancia
        matches = sorted(matches, key=lambda x: x.distance)
        coincidencias.append(matches[i])

        # Extraigo los keypoints de la imagen i que están en correspondencia con los keypoints de la imagen i+1
        keypoints_imagen1 = np.float32([keypoints[i][j.queryIdx] for j in matches])
        keypoints_imagen2 = np.float32([keypoints[i+1][j.trainIdx] for j in matches])

        # Calcular la homografía entre los dos conjuntos de puntos
        h, status = (cv2.findHomography(srcPoints=keypoints_imagen1, dstPoints=keypoints_imagen2, method=cv2.RANSAC, ransacReprojThreshold=1))
        homografia.append(h)

        # Borramos el contenido de dichos keypoints para usarlos la siguiente iteracion
        np.array([row for row in keypoints_imagen1 if len(row)<=3])
        np.array([row for row in keypoints_imagen2 if len(row)<=3])

    # Nos creamos un fondo negro, con un tamaño específico (para que quepan las demás fotografías)
    ancho = lista_imagenes[0].shape[0]*6
    alto = lista_imagenes[0].shape[1]*4

    # Obtenemos la imagen del centro
    centro = len(lista_imagenes) // 5

    # Definimos la traslacion que nos pone la imagen central del mosaico en el centro
    tras = np.matrix([[1, 0, lista_imagenes[centro].shape[1]], [0, 1, lista_imagenes[centro].shape[0]], [0, 0, 1]], dtype=np.float32)
    # Llevamos esa imagen al centro de nuestro mosaico con la homografia
    mosaico = cv2.warpPerspective(src=lista_imagenes[centro], M=tras, dsize=(ancho, alto), borderMode=cv2.BORDER_TRANSPARENT)

    # Calculamos las homografias que se le aplican a las imagenes de la izquierda de la imagen central
    for i in range(0, centro):

        # Definimos la traslacion para las imágenes de la izquierda
        izquierda = np.matrix([[1, 0, 1],[0, 1, 1],[0, 0, 1]], dtype=np.float32)

        for j in range(i, centro): izquierda = homografia[j] * izquierda

        # Las llevamos al mosaico
        cv2.warpPerspective(src=lista_imagenes[i], M=tras*izquierda, dst=mosaico, dsize=(ancho, alto), borderMode=cv2.BORDER_TRANSPARENT)

    # Calculamos las homografias que se le aplican a las imagenes de la derecha de la imagen central
    # Ahora debemos usar las inversas de las homografías, ya que las homografias que se usan son de la imagen i a la i-1
    for i in range(centro + 1, len(lista_imagenes)):

        # Definimos la traslacion para las imágenes de la derecha
        derecha = np.matrix([[1, 0, 1], [0, 1, 1], [0, 0, 1]], dtype=np.float32)

        for j in range(centro, i): derecha = derecha * np.linalg.inv(homografia[j])

        # Las llevamos al mosaico
        cv2.warpPerspective(lista_imagenes[i], M=tras*derecha, dst=mosaico, dsize=(ancho, alto),borderMode=cv2.BORDER_TRANSPARENT)

    return mosaico


# _#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#
#                                                       BONUS 3                                                        #
# _#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#


'''
# Vamos a crear una función con la que obtengamos la homografia si existen 4 correspondencias
# entre las imágenes
def obtenerHomografia(correspondencias):
    
    lista = []

    # Recorremos las correspondencias
    for i in correspondencias:

    	# Obtenemos los puntos 
        x = np.matrix([i[0], i[1], 1])
        y = np.matrix([i[2], i[3], 1])

        # Apuntes teoría Tema 3 (página 48) --> Solving for Homographies
        ecuacion1 = [-y[2]*x[0], -y[2]*x[1], -y[2]*x[2], 0, 0, 0, y[0]*x[0], y[0]*x[1], y[0]*x[2]]
        ecuacion2 = [0, 0, 0, -y[2]*x[0], -y[2]*x[1], -y[2]*x[2], y[1]*p1[0], y[1]*x[1], y[1]*x[2]]
        
        lista.append(ecuacion1, ecuacion2)
     
    # Convertimos la lista en en una matriz
    matriz = np.matrix(lista)

    # Hacemos la composición SVD
    u,s,v = np.linalg.svd(matriz)

    # Cogemos el minimo valor de 
    h = np.reshape(np.min(v), (3, 3))

    return h
    
# Algoritmo Ransac, creamos la homogbrafia a partir de las
# correspondencias de la anterior función
def ransac(correspondencias, umbral):
    
    # Variables a usar en el algorimot
    inliers = []

    # Obtenemos la longitud de las correspondencias
    len_correspondencias = len(correspondencias)

    # Establecemos un número de iteraciones
    for i in range(100):
        
        # Obtenemos 4 valores aleatorios de las correspondencias
        correspondencia1 = correspondencias[randrange (0, 
                                            len_correspondencias)]
        correspondencia2 = correspondencias[randrange (0, 
                                             len_correspondencias)]
        correspondencia3 = correspondencias[randrange (0, 
                                             len_correspondencias)]
        correspondencia4 = correspondencias[randrange (0, 
                                            len_correspondencias)]

        # Juntamos las correspondencias por columnas
        resultado = np.vstack((correspondencia1, correspondencia2))
        resultado = np.vstack((resultado, correspondencia3))
        resultado = np.vstack((resultado, correspondencia4))

        # Obtener homografia de esos puntos
        H = obtenerHomografia(resultado)
        
	# Obtener el mayor conjunt de inliers 
        inliers = []
        for i in range(len_correspondencias):
            inliers = (correspondencias [i])
            i.append(inliers)

    return H, inliers

    

'''

if __name__ == '__main__':

    image1 = leer_imagen(filename='imagenes/Tablero1.jpg', flag_color=False)
    image2 = leer_imagen(filename='imagenes/Tablero2.jpg', flag_color=False)
    image3 = leer_imagen(filename='imagenes/yosemite_full/Yosemite1.jpg', flag_color=False)
    image4 = leer_imagen(filename='imagenes/yosemite_full/Yosemite2.jpg', flag_color=False)
    image5 = leer_imagen(filename='imagenes/yosemite_full/Yosemite3.jpg', flag_color=False)
    image6 = leer_imagen(filename='imagenes/yosemite_full/Yosemite4.jpg', flag_color=False)
    image7 = leer_imagen(filename='imagenes/yosemite_full/Yosemite5.jpg', flag_color=False)
    image8 = leer_imagen(filename='imagenes/yosemite_full/Yosemite6.jpg', flag_color=False)
    image9 = leer_imagen(filename='imagenes/yosemite_full/Yosemite7.jpg', flag_color=False)

    imageA = cv2.imread('imagenes/mosaico-1/mosaico002.jpg', flags=1)
    imageB = cv2.imread('imagenes/mosaico-1/mosaico003.jpg', flags=1)
    imageC = cv2.imread('imagenes/mosaico-1/mosaico004.jpg', flags=1)
    imageD = cv2.imread('imagenes/mosaico-1/mosaico005.jpg', flags=1)
    imageE = cv2.imread('imagenes/mosaico-1/mosaico006.jpg', flags=1)
    imageF = cv2.imread('imagenes/mosaico-1/mosaico007.jpg', flags=1)
    imageG = cv2.imread('imagenes/mosaico-1/mosaico008.jpg', flags=1)
    imageH = cv2.imread('imagenes/mosaico-1/mosaico009.jpg', flags=1)
    imageI = cv2.imread('imagenes/mosaico-1/mosaico010.jpg', flags=1)
    imageJ = cv2.imread('imagenes/mosaico-1/mosaico011.jpg', flags=1)

    # ------------------------------------------------------------------------------------------------------------------

    # Ejercicio 1 #

    # Apartado A - B - C

    # _____________ PARA LA IMAGEN 1 (TABLERO) _____________ #

    # Creamos la pirámide gaussiana de 6 niveles (imagen original + 5 niveles)
    niveles_piramide1, piramide1 = piramide_gaussiana(imagen=image1, levels=6, borde=cv2.BORDER_DEFAULT)

    # Obtenemos los puntos de Harris en nuestra imagen y la imagen binaria, y representamos la image
    puntos_harris1, imagen_binaria1 = obtenerPuntosHarris(lista_images=niveles_piramide1, block_size=5, k_size=5, umbral=0.2)

    # Aplicamos supresión de no máximos y representamos la imagen
    imagen_binaria_supresion1 = supresionNoMaximos(puntos_harris=puntos_harris1, imagen_binaria=imagen_binaria1, tam_ventana=3)
    representar_imagenes([puntos_harris1[0], imagen_binaria1[1], imagen_binaria_supresion1[0]],
                         ['Puntos de Harris', 'En Imagen Binaria', 'Con Supresión de No Maximos'])

    # Nos quedamos con los mejores puntos
    mejores_puntos1 = seleccionar500MejoresPuntosHarris(puntos_harris=puntos_harris1, imagen_binaria=imagen_binaria_supresion1, tam_ventana=3)
    dibujarCirculoLineas(imagen=image1, puntos_harris=mejores_puntos1, titulo="Tablero Puntos Harris")

    # Obtenemos los puntos refinados
    puntos_refinados1 = refinarPuntosHarris(lista_imagenes=niveles_piramide1, puntos_harris=mejores_puntos1)
    #dibujarCirculoLineas(imagen=image1, puntos_harris = puntos_refinados1, titulo="Tablero Puntos Refinados")

    # Obtenemos la orientación de cada punto
    orientaciones1 = obtenerOrientacionPunto(lista_imagenes=niveles_piramide1, puntos_refinados=puntos_refinados1)
    #dibujarCirculoLineas(imagen=image1, puntos_harris=puntos_refinados1, titulo="Tablero Puntos Orientaciones", usarOrientaciones=True, orientaciones=orientaciones1)


    # _____________ PARA LA IMAGEN 3 (YOSEMITE) _____________ #

    # Creamos la pirámide gaussiana de 6 niveles (imagen original + 5 niveles)
    niveles_piramide3, piramide3 = piramide_gaussiana(imagen=image3, levels=6, borde=cv2.BORDER_DEFAULT)

    # Obtenemos los puntos de Harris en nuestra imagen y la imagen binaria, y representamos la image
    puntos_harris3, imagen_binaria3 = obtenerPuntosHarris(lista_images=niveles_piramide3, block_size=5, k_size=5, umbral=0.012)

    # Aplicamos supresión de no máximos y representamos la imagen
    imagen_binaria_supresion3 = supresionNoMaximos(imagen_binaria=imagen_binaria3, puntos_harris=puntos_harris3, tam_ventana=3)
    representar_imagenes([puntos_harris3[0], imagen_binaria3[1], imagen_binaria_supresion3[0]],
                         ['Puntos de Harris', 'En Imagen Binaria', 'Con Supresión de No Maximos'])

    # Nos quedamos con los mejores puntos
    mejores_puntos3 = seleccionar500MejoresPuntosHarris(puntos_harris=puntos_harris3, imagen_binaria=imagen_binaria_supresion3, tam_ventana=3)
    dibujarCirculoLineas(imagen=image3, puntos_harris=mejores_puntos3, titulo="Yosemite Puntos Harris")

    # Obtenemos los puntos refinados
    puntos_refinados3 = refinarPuntosHarris(lista_imagenes=niveles_piramide3, puntos_harris=mejores_puntos3)
    dibujarCirculoLineas(imagen=image3, puntos_harris=puntos_refinados3, titulo="Yosemite Puntos Refinados")

    # Obtenemos la orientación de cada punto
    orientaciones3 = obtenerOrientacionPunto(lista_imagenes=niveles_piramide3,puntos_refinados=puntos_refinados3)
    dibujarCirculoLineas(imagen=image3, puntos_harris=puntos_refinados3, titulo="Yosemite Puntos Orientaciones", usarOrientaciones=True, orientaciones=orientaciones3)

    # Apartado D
    image3 = leer_imagen(filename='imagenes/yosemite_full/Yosemite1.jpg', flag_color=False)
    keypoints(puntos_refinados3, image3)

    # ------------------------------------------------------------------------------------------------------------------

    # Ejercicio 2 #

    # Yosemite 1 con Yosemite 2
    establecerCorrespondenciasFuerzaBruta(image3, image4)
    # Yosemite 5 con Yosemite 6
    establecerCorrespondenciasFuerzaBruta(image7, image8)

    # Yosemite 1 con Yosemite 2
    establecerCorrespondencias2NN(image3, image4)
    # Yosemite 5 con Yosemite 6
    establecerCorrespondencias2NN(image7, image8)

    # Con mis Keypoints
    image1 = leer_imagen(filename='imagenes/Tablero1.jpg', flag_color=False)
    (keypoints1,descriptor1) = keypoints(puntos_refinados1, image1)
    (keypoints2,descriptor2) = keypoints(puntos_refinados1, image1)
    correspondenciasMisKeyPoints(keypoints1, image1, descriptor1, keypoints2, image1, descriptor2)

    # ------------------------------------------------------------------------------------------------------------------

    # Ejercicio 3 #
    resultado = mosaicoN3(imageA=imageA, imageB=imageC, imageC=imageB)
    #cv2.imwrite("imagenes/mosaico3.jpg", resultado)
    representar_imagenes([imageA, imageB, imageC, resultado],["Imagen 1", "Imagen 2", "Imagen 3", "Mosaico 3"])

    # ------------------------------------------------------------------------------------------------------------------

    # Ejercicio 4 #

    mosaico = mosaicoN([imageA, imageB, imageC, imageD, imageE, imageF, imageG, imageH, imageI, imageJ])
    #cv2.imwrite("imagenes/mosaico.jpg", mosaico)
    representar_imagenes([mosaico],['Mosaico'])

    # -----------------------------------------------------------------------------------------------------------------

    # BONUS 3 #

