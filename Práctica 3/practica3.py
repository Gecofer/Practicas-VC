#! /usr/bin/env python
# -*- coding: utf-8 -*-

#######################################################################################################################
# Trabajo 3: Indexación y Recuperación de Imágenes                                                                    #
# Curso 2017/2018                                                                                                     #
# Gema Correa Fernández                                                                                               #
#######################################################################################################################

# Importamos las librerías necesarias para desarrollar la práctica
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import collections
from funciones_auxiliares import *
from auxFunc import *

# -------------------------------------------------------------------------------------------------------------------- #
#                                                       EJERCICIO 1                                                    #
# -------------------------------------------------------------------------------------------------------------------- #

'''
Leer parejas de imágenes que tengan partes de escena comunes. Seleccionar una región en la primera imagen que esté 
presente en la segunda imagen y extraer los puntos SIFT contenidos en la región seleccionada de la primera imagen. 
Calcular las correspondencias con todos los puntos SIFT de la segunda imagen. Pintar las correspondencias encontradas 
sobre las imágenes.
'''

def emparejamientoDescriptores(imagen1, imagen2, puntos):

    '''
    :param imagen1: imagen de la que extraigo una región
    :param imagen2: imagen tiene presente la región extraída de la imagen1
    :param puntos: región de la primera imagen
    :return:
    '''

    # Inicializamos una máscara con las dimensiones de la imagen
    mask = np.zeros((imagen1.shape[0], imagen1.shape[1]), dtype=np.uint8)

    # Crear la máscara que define el polígono de puntos (región que hemos seleccionado)
    cv2.fillConvexPoly(mask, puntos, 1)

    # Usamos el descriptor SIFT
    sift = cv2.xfeatures2d.SIFT_create()

    # Extraigo los puntos SIFT contenidos en la región seleccionada de la primera imagen
    keypoints_imagen1, descriptors_imagen1 = sift.detectAndCompute(image=imagen1, mask=mask) # (le paso la mascara)

    # Extraigo todos los puntos SIFT de la segunda imagen
    keypoints_imagen2, descriptors_imagen2 = sift.detectAndCompute(image=imagen2, mask=None)

    # Calculo las correspondencias con todos los puntos SIFT de la segunda imagen
    correspondecias = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)
    matches = correspondecias.match(descriptors_imagen1, descriptors_imagen2)

    # Ordenamos las correspondencias por mejor distancia
    matches = sorted(matches, key=lambda x: x.distance)

    # Cogemos un numero de correspondecias extraidas para apreciar la calidad de los resultados
    numero_correspondencias = 2

    # Pinto las correspondencias encontrados sobre las imágenes.
    resultado = cv2.drawMatches(img1=imagen1, keypoints1=keypoints_imagen1, img2=imagen2, keypoints2=keypoints_imagen2,
                                    matches1to2=matches[:numero_correspondencias], outImg=imagen2.copy(), flags=0)

    representar_imagenes([resultado],['Emparejamiento de Descriptores'])
    #cv2.imwrite("Salida/ejercicio1.png", resultado)



# -------------------------------------------------------------------------------------------------------------------- #
#                                                       EJERCICIO 2                                                    #
# -------------------------------------------------------------------------------------------------------------------- #

'''
Usando las imágenes dadas en imagenesIR.rar se han extraido 600 regiones de cada imagen y se ha construido un 
vocabulario de 5.000 palabras usando k-means. Se han extraido de forma directa las regiones imágenes asociadas y se 
han re-escalado a 24x24 píxeles Los ficheros con los datos son descriptors.pkl, vocabulary.pkl y patches.pkl. Leer 
los ficheros usando loadAux() y loadDictionary(). Elegir al menos dos palabras visuales diferentes y visualizar las 
regiones imagen de los 20 parches más cercanos de cada palabra visual, de forma que se muestre el contenido visual que 
codifican. Explicar lo que se ha obtenido.

Hacer clustering y sacar los clusters. Luego coger los clusters con menor varianza (que se supone que son más 
representativos) y para cada uno de esos clusters que coges, pintas los patches asociados a los mejores descriptores 
(que son los que están más cercanos al centroide del cluster)
'''

def visualizacionVocabulario ():

    titulos = []
    imagenes = []
    var_centroide = []
    varianza_final = []
    var_descriptors = []
    descriptors_labels = []
    varianza_descriptors = []
    descriptors_centroide = []
    distancia_descriptor_labels = []
    descriptors_centroide1_mejores = []
    descriptors_centroide2_mejores = []

    # Defino el tamaño que de los labels, descriptores son 1-1 (193041)
    tam = 10892

    # Cargamos el diccionario
    accuracy, labels, dictionary = loadDictionary("kmeanscenters5000.pkl")

    # Convierto los labels en una lista
    labels_lista=[]
    for x in labels:
        for y in x:
            labels_lista.append(y)

    # Cargamos los descriptores y sus parches extraidos de las imágenes
    descriptors, patches = loadAux("descriptorsAndpatches.pkl", True)

    # Como los descriptores no están normalizados debemos normalizarlos
    for i in range(0,descriptors.shape[0]):
        # Aplicamos la norma: elevar al cuadrado cada elemento del descriptor, sumarlos y aplicarles la raíz cuadrada
        norm = np.sqrt(np.sum(descriptors[i]*descriptors[i]))
        # Normalizar el descriptor
        descriptors[i] = descriptors[i]/norm

    # Agrupamos los descriptores en función del centroide al que esten asociados, a partir de las etiquetas (labels)
    # Las etiquetas y los descriptores se corresponden 1-1, la primera etiqueta representa el primer descriptor
    # y el centroide al que esta asociado (contenido del label es un numero del 0 al 4999)
    for i,j in zip(labels_lista[:tam], descriptors[:tam]):
        descriptors_centroide_aux = i,j
        descriptors_centroide.append(descriptors_centroide_aux)

    # Calculo la distancia del descriptor a su centroide asociado, para ello hago la resta entre el descriptor en el
    # que estoy con su respectiva etiqueta (guardo la distancia)
    for i in range(0, tam): # len(descriptors)
        # np.linalg.norm(descriptor_en_cuestion - centroide_asociado)
        distancia_descriptor_labels_aux = np.linalg.norm(descriptors[i] - labels[i])
        distancia_descriptor_labels.append(distancia_descriptor_labels_aux)

    # Asocio la distancia del descriptor a su centroide asociado
    for i,j in zip(labels_lista[:tam], distancia_descriptor_labels[:tam]):
        descriptors_labels_aux = i,j
        descriptors_labels.append(descriptors_labels_aux)

    # Ordenar estos descriptores en función de su distancia al centroide asociado (de menor a maypr)
    descriptors_labels = sorted(descriptors_labels, key=lambda a_entry: a_entry[1])

    # Me quedo con los 20 más cercanos, que serán los que más cerca se encuentren
    descriptors_mas_cercanos = descriptors_labels[:20]

    # Una vez que tengo los 20 descriptores mas cercanos para cada centroide, calculo la varianza entre los descriptores
    # Cojo las mejores varianzas y de esos centroides imprimes los parches asociados
    # Una varianza elevada significa que los datos están más dispersos, mientras que un valor de varianza bajo indica
    # que los valores están por lo general más próximos a la media
    for i in descriptors_mas_cercanos:
        for j in descriptors_centroide[:tam]:
            if i[0] == j[0]:
                varianza_descriptors.append(j)

    # Asocio la varianza del descriptor con su centroide asociado
    for i in range(len(varianza_descriptors)):
        var_centroide_aux = varianza_descriptors[i][0]; var_centroide.append(var_centroide_aux)
        var_descriptors_aux = np.mean(np.var(varianza_descriptors[i][1])); var_descriptors.append(var_descriptors_aux)

    for i,j in zip(var_centroide, var_descriptors):
        varianza_final_aux = i,j; varianza_final.append(varianza_final_aux)

    # Ordenamos las varianza de menor a valor valor (cogemos las mejores varianzas)
    varianza_final = sorted(varianza_final, key=lambda a_entry: a_entry[1])

    '''
    Elegir al menos dos palabras visuales diferentes y visualizar las 
    regiones imagen de los 20 parches más cercanos de cada palabra visual, de forma que se muestre el contenido visual que 
    codifican.
    '''
    # Coger los dos centroides en los que he tenido una menor varianza en sus descriptores

    # Elijo un centroide e imprimo los parches asociados
    centroide1 = varianza_final[63][0] # 44 buena (56)

    # Para el centroide 1 obtengo los indices de los descriptores e imprimo los 20 parches asociados a ellos
    for i in varianza_descriptors:
        if i[0] == centroide1: descriptors_centroide1 = i[1]

    # Cogemos los 20 mejores
    descriptors_centroide1 = np.argsort(descriptors_centroide1)[:15]

    for i in descriptors_centroide1:
        descriptors_centroide1_mejores.append(i)

    # descriptors_centroide1 = [29, 99, 100, 20, 89, 50, 28, 101, 12, 114, 62, 109, 59, 106, 103, ]

    # Visualizar las regiones imagen de los 20 parches más cercanos a cada palabra
    for i in descriptors_centroide1_mejores:
        imagenes.append(patches[i])
        titulos.append("")

    representar_imagenes(imagenes,titulos)

    # Elijo un centroide e imprimo los parches asociados
    centroide2 = varianza_final[0][0]

    # Para el centroide 1 obtengo los indices de los descriptores e imprimo los 20 parches asociados a ellos
    for i in varianza_descriptors:
        if i[0] == centroide2: descriptors_centroide2 = i[1]

    # Cogemos los 20 mejores
    descriptors_centroide2 = np.argsort(descriptors_centroide2)[:20]

    for i in descriptors_centroide2:
        descriptors_centroide2_mejores.append(i)

    # descriptors_centroide1 = [29, 99, 100, 20, 89, 50, 28, 101, 12, 114, 62, 109, 59, 106, 103, ]
    imagenes = []
    titulos = []
    # Visualizar las regiones imagen de los 20 parches más cercanos a cada palabra
    for i in descriptors_centroide2_mejores:
        imagenes.append(patches[i])
        titulos.append("")

    representar_imagenes(imagenes,titulos)



# -------------------------------------------------------------------------------------------------------------------- #
#                                                       EJERCICIO 3                                                    #
# -------------------------------------------------------------------------------------------------------------------- #

'''
Implementar un modelo de índice invertido + bolsa de palabras para las imágenes dadas en imágenesIR.rar usando 
el vocabulario calculado en el punto anterior. Verificar que el modelo construido para cada imagen permite recuperar 
imágenes de la misma escena cuando la comparamos al resto de imágenes de la base de datos. Elegir dos imágenes-pregunta 
en las se ponga de manifiesto que el modelo usado es realmente muy efectivo para extraer sus semejantes y elegir otra 
imagen-pregunta en la que se muestre que el modelo puede realmente fallar. Para ello muestre las cinco imágenes más 
semejantes de cada una de las imágenes-pregunta seleccionadas usando como medida de distancia el producto escalar 
normalizado de sus vectores de bolsa de palabras. Explicar los resultados.
'''

def recuperacionImagenes(image_test):

    '''
    :param image_test: imagen pregunta
    :return: devuelve la imagen pregunta, con las 5 imágenes más semejantes
    '''

    cont = 0
    imagen_final = []
    lista_titulos = []
    lista_imagenes = []
    mostrar_imagenes = []
    fichero_invertido = []
    norma_diccionario = []
    indices_image_test = []
    recuperar_imagenes = []
    histograma_imagenes = []
    palabras_image_test = []

    # Cargamos el diccionario (el diccionario ya está normalizado)
    accuracy, labels, dictionary = loadDictionary("kmeanscenters5000.pkl")

    # Normalizamos el diccionario como pone en la fórmula de la página 17 del tema "13.instances_recog.pdf"
    # Como el descriptor va a ser el mismo por columnas, no hace falta normalizarlo, ya que estaría diviendo y
    # multiplicando siempre por el mismo número
    for i in range (0, dictionary.shape[0]):
        norma_diccionario.append(np.sqrt(np.sum(dictionary[i]*dictionary[i])))

    # Nuestra norma_diccionario es una lista con 5000 elementos, lo convierto en vector
    norma_diccionario = np.asarray(norma_diccionario, dtype=np.float32)

    # Recorremos las 441 imágenes que tenemos en "imagenesIR.rar"
    for i in range(0,441):

        # Leemos todas las imagenes
        name = "imagenesIR/" + str(i) + ".png"
        image = cv2.imread(name)

        # Extraemos los descriptores SIFT de todas las imágenes
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints_images, descriptors_images = sift.detectAndCompute(image=image, mask=None)

        # Hacemos el denominador de la fórmula de la semejanza {𝑑𝑗,𝑞} (pagina 17 en "13.instances_recog.pdf")
        # Obtengo una matriz donde cada fila es una palabra y cada columna es un descriptor (5000 x descriptores)
        # Transpongo los descriptores para hacer la multiplicacion
        # dicionario: (5000, 128); descriptor: (n_descriptores, 128)
        semejanza_denominador = np.dot(dictionary, descriptors_images.T)

        # Calculo la fórmula de la semejanza ({𝑑𝑗,𝑞}) / (||𝑑𝑗|| ||𝑞||)
        semejanza_imagenes = np.divide(semejanza_denominador, norma_diccionario[:,None])

        # Una vez calculada mi matriz de semejanzas, busco el máximo para cada columna (relacionar)
        relaciones_imagenes = np.argmax(semejanza_imagenes, axis=0)

        # Calculo el histograma para cada palabra y veo sus votos con "np.bincount"
        histograma = np.bincount(relaciones_imagenes, weights=None, minlength=5000)
        histograma_imagenes.append(histograma)

    # Convierto en array "histograma_imagenes"
    histograma_imagenes = np.array(histograma_imagenes)

    # Construyo el fichero invertido (sacar de cada histograma a que imagen va)
    # Por ejemplo, la imagen 0 tiene (palabras columnas). Cogemos de cada palabra las imágenes que tienen voto distinto de 0
    for i in range (0,5000):
        palabras = np.where(histograma_imagenes[:,i] != 0)[0]
        fichero_invertido.append(palabras)

    # Ya tengo para cada palabra asociado su lista de imágenes (fichero_invertido)
    # Ahora paso a realizar el mismo proceso que antes, pero para la imagen-pregunta leer la imagen test, la de consulta

    # Extraemos los descriptores SIFT de la imagen_pregunta
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_image_test, descriptors_image_test = sift.detectAndCompute(image=image_test, mask=None)

    # Hacemos el denominador de la fórmula de la semejanza {𝑑𝑗,𝑞} (pagina 17 en "13.instances_recog.pdf")
    # Obtengo una matriz donde cada fila es una palabra y cada columna es un descriptor (5000 x descriptores)
    # Transpongo los descriptores para hacer la multiplicacion
    # dicionario: (5000, 128); descriptor: (n_descriptores, 128)
    semejanza_denominador_test = np.dot(dictionary, descriptors_image_test.T)

    # Calculo la fórmula de la semejanza ({𝑑𝑗,𝑞}) / (||𝑑𝑗|| ||𝑞||)
    semejanza_image_test = np.divide(semejanza_denominador_test, norma_diccionario[:,None])

    # Una vez calculada mi matriz de semejanzas, busco el máximo para cada columna (relacionar)
    relacion_image_test = np.argmax(semejanza_image_test, axis=0)

    # Calculo el histograma para cada palabra y veo sus votos con "np.bincount"
    histograma_test = np.bincount(relacion_image_test, weights=None, minlength=5000)

    # Cojo en el histograma los votos que tienen un valor distinto de 0
    for i in range(0,5000):
        palabras_image_test.append(np.where(histograma_test[i] != 0)[0])

    # Cojo los índices de las imágenes de mi histograma test
    for i in histograma_test:
        if i != 0: indices_image_test.append(cont)
        cont = cont+1

    # Una vez esto, vamos a seleccionar las cinco imágenes más semejantes de cada una de las imágenes-pregunta
    # Tengo creado mi histograma_test. Así que voy al fichero_invertido y recupero las imágenes correspondiente a
    # las palabras de mi histograma
    for i in indices_image_test:
        recuperar_imagenes.append(fichero_invertido[i])

    # Obtengo la lista de imágenes
    recuperar_imagenes = np.concatenate(recuperar_imagenes)

    # Nos quedamos con las imágenes más repetidas
    imagenes_repetidas = np.unique(recuperar_imagenes, return_counts=True)
    imagenes_repetidas_id = imagenes_repetidas[0]
    imagenes_repetidas_contador = imagenes_repetidas[1]

    for i in range(0,441):

        # cogo las imagenes que mas se repiten
        if (len(imagenes_repetidas_contador) != 0) and (len(imagenes_repetidas_id) != 0):

            # obtengo el indice del valor max
            index = np.argmax(imagenes_repetidas_contador)

            # lo obtengo de la anterior lista
            valor = imagenes_repetidas_id[index]

            # lo borro de ambas listas
            imagenes_repetidas_contador = np.delete(imagenes_repetidas_contador, index)
            imagenes_repetidas_id = np.delete(imagenes_repetidas_id, index)

            imagen_final.append(valor)

    # Para cada imagen de la lista de imagenes comparo histogramas_imagenes con histograma_imagen_test
    # Comparo las semejanzas entre ellas, para escojo un umbral
    for i in imagen_final:
        comparar_histogramas = np.dot(histograma_imagenes[i], histograma_test)
        #print(comparar_histogramas)
        if comparar_histogramas > 900: mostrar_imagenes.append(i)

    # Me quedo con las 5 imagenes semejantes
    imagenes_finales = mostrar_imagenes[:5]

    # Vamos a visualizar las imágenes
    # Metemos la imagen TEST
    lista_imagenes.append(image_test)
    lista_titulos.append("Original")

    for i in imagenes_finales:
        nombre = "imagenesIR/" + str(i) + ".png"
        img = cv2.imread(nombre,1)
        lista_imagenes.append(img)
        lista_titulos.append("")

    # Representamos las imagenes
    representar_imagenes(lista_imagenes,lista_titulos)



if __name__ == '__main__':

    # Ejercicio 1

    # Leemos las parejas de imágenes
    pareja1_1 = leer_imagen(filename='imagenesIR/15.png', flag_color=False)
    pareja1_2 = leer_imagen(filename='imagenesIR/16.png', flag_color=False)

    pareja2_1 = leer_imagen(filename='imagenesIR/337.png', flag_color=False)
    pareja2_2 = leer_imagen(filename='imagenesIR/338.png', flag_color=False)

    pareja3_1 = leer_imagen(filename='imagenesIR/390.png', flag_color=False)
    pareja3_2 = leer_imagen(filename='imagenesIR/391.png', flag_color=False)

    pareja4_1 = leer_imagen(filename='imagenesIR/44.png', flag_color=False)
    pareja4_2 = leer_imagen(filename='imagenesIR/49.png', flag_color=False)

    pareja5_1 = leer_imagen(filename='imagenesIR/243.png', flag_color=False)
    pareja5_2 = leer_imagen(filename='imagenesIR/246.png', flag_color=False)

    # Selecciono una región de cada pareja de imágenes y extraigo sus puntos que me interesan
    #extractRegion(pareja1_1)
    #extractRegion(pareja2_1)
    #extractRegion(pareja3_1)
    #extractRegion(pareja4_1)
    #extractRegion(pareja5_1)
    pareja1_puntos = np.array([[161, 211],[227, 204],[235, 258],[227, 266],[227, 300],[174, 306],[159, 212]])
    pareja2_puntos = np.array([[208, 387], [347, 394],[341, 445],[206, 442],[200, 391]])
    pareja3_puntos = np.array([[113, 155],[140, 158],[149, 196],[145, 205],[117, 201],[112, 155]])
    pareja4_puntos = np.array([[156, 180],[80, 250],[93, 421],[95, 466],[159, 469],[123, 294],[127, 233],[154, 180]])
    pareja5_puntos = np.array([[226, 108],[242, 77],[264, 90],[256, 128],[232, 156],[224, 113]])

    emparejamientoDescriptores(pareja1_1, pareja1_2, pareja1_puntos) # Pareja 1
    emparejamientoDescriptores(pareja2_1, pareja2_2, pareja2_puntos) # Pareja 2
    emparejamientoDescriptores(pareja3_1, pareja3_2, pareja3_puntos) # Pareja 3
    emparejamientoDescriptores(pareja4_1, pareja4_2, pareja4_puntos) # Pareja 4
    emparejamientoDescriptores(pareja5_1, pareja5_2, pareja5_puntos) # Pareja 5

    # ------------------------------------------------------------------------------------------------------------------

    # Ejercicio 2

    visualizacionVocabulario()

    # ------------------------------------------------------------------------------------------------------------------

    # Ejercicio 3

    # Imagen 86 y umbral 900 (obtenemos una salida buena)
    # Imagen 333 y umbral 400 (obtenemos una salida buena)
    # Imagen 366 y umbral 289 (obtenemos una salida con 4 iguales y otra distinta pero con el mismo color)
    # Imagen 202 y umbral 289 (obtenemos una salida con 4 iguales y otra distinta pero con el mismo color)
    # Imagen 36 y umbral 150 (mala)
    # Imagen 62 y umbral 300 (mala)
    image_test = cv2.imread("imagenesIR/89.png",1)

    recuperacionImagenes(image_test)
