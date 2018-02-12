#! /usr/bin/env python
# -*- coding: utf-8 -*-


##########################################################################################################
# Trabajo 1: Filtrado y Muestreo
# Curso 2017/2018
# Gema Correa Fernández

##########################################################################################################


# Importamos las librerías necesarias para desarrollar la práctica
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math


# ------------------------------------------------------------------------------------------------------- #
# EJERCICIO 1
# ------------------------------------------------------------------------------------------------------- #

## APARTADO A ##

# Función que lee una imagen (leemos una imagen con la funcion de opencv 'imread' con el flag especificado)
# Entrada:
#  - filename: nombre del archivo desde el que se lee la imagen
#  - flag_color: variable booleana que nos indica si leemos una imagen en color o en escala de grises
# Salida: devuelve la matriz (el objeto) donde se almacena
def leer_imagen(filename, flag_color=True):

	if (flag_color): # Si flag_color es True, carga una imagen en color de 3 canales
		image = cv2.imread(filename, flags=1);

	else: # Si flag_color es False, carga una imagen en escala de grises
		image = cv2.imread(filename, flags=0);

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
		return -1 # No hay el mismo numero de imágenes que de títulos

	# Calculamos el numero de imagenes
	n_imagenes = len(lista_imagen_leida)

	# Establecemos por defecto el numero de columnas
	n_columnas = 2

	# Calculamos el número de filas
	n_filas = (n_imagenes // n_columnas) + (n_imagenes % n_columnas)

	# Establecemos por defecto un tamaño a las imágenes
	plt.figure(figsize=(7,7))

	# Recorremos la lista de imágenes
	for i in range(0, n_imagenes):

		plt.subplot(n_filas, n_columnas, i+1) # plt.subplot empieza en 1

		if (len(np.shape(lista_imagen_leida[i]))) == 2: # Si la imagen es en gris
			plt.imshow(lista_imagen_leida[i], cmap = 'gray')
		else: # Si la imagen es en color
			plt.imshow(cv2.cvtColor(lista_imagen_leida[i], cv2.COLOR_BGR2RGB))

		plt.title(lista_titulos[i]) # Añadimos el título a cada imagen

		plt.xticks([]), plt.yticks([]) # Para ocultar los valores de tick en los ejes X e Y

	plt.show()


## APARTADO B ##

# Función que desenfoca una imagen usando un filtro gaussiana
# Entrada:
#  - imagen: nombre del archivo al cual se le va a aplicar el filtro
#  - sigma: valor del sigma
#  - borde: borde que queremos aplicar a la imagen
# Salida: devuelve la imagen con el filtro aplicado
def alisar_imagen(imagen, sigma, borde):

	# Calculamos el tamanio de la mascara
	tamanio = 3*sigma * 2 + 1

	# Aplicamos el filtro a la imagen
	blur = cv2.GaussianBlur(imagen, ksize=(tamanio,tamanio),
							sigmaX = sigma, sigmaY=sigma, borderType=borde)


	# Cambiamos la imagen a una profundidad(color) de 8 o 16 bits sin signo int (8U, 16U)
	# o 32 bit flotante (32F)
	return blur.astype(np.uint8)


## APARTADO C ##

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
				fila = cv2.filter2D(imagen[i,:], -1, mask_fila, borderType=borde)

				# debemos poner fila[:,0] ya que filter2D crea una estructura de canales
				# pero la imagen no tiene canales
				nueva_imagen[i,:] = fila[:,0]

		if mask_col is not None:

			nueva_imagen = nueva_imagen.transpose(1,0) # Transponemos para cambiar filas por columnas

			# Iteramos ahora por columnas (igual que con las filas)
			for i in range(np.shape(imagen)[1]):

				# Aplicamos convolución
				columna = cv2.filter2D(nueva_imagen[i,:], -1, mask_col, borderType=borde)
				nueva_imagen[i,:] = columna[:,0]

			# Volvemos a trasponer para obtener la imagen original alisada
			nueva_imagen = nueva_imagen.transpose(1,0)

	else: # Imagen en color

		if mask_fila is not None:

			# Iteramos filas
			for i in range(np.shape(imagen)[0]):
				# Aplicamos convolución por cada canal
				fila = cv2.filter2D(imagen[i,:,:], -1, mask_fila, borderType=borde)
				nueva_imagen[i,:,:] = fila

		if mask_col is not None:

			nueva_imagen = nueva_imagen.transpose(1,0,2) # Transponemos para cambiar filas por columnas

			# Iteramos ahora por columnas
			for i in range(np.shape(imagen)[1]):
				# Aplicamos convolución por cada canal
				columna = cv2.filter2D(nueva_imagen[i,:,:], -1, mask_col, borderType=borde)
				nueva_imagen[i,:,:] = columna

			# Volvemos a trasponer para obtener la imagen original alisada
			nueva_imagen = nueva_imagen.transpose(1,0,2)

	return nueva_imagen.astype(np.uint8)


## APARTADO D ##

# Función de convolución con núcleo de primera derivada
# Entrada:
#  - imagen: nombre del archivo al cual se le va a aplicar el filtro
#  - sigma: valor del sigma
#  - borde: borde que queremos aplicar a la imagen
# Salida: visualización de imágenes aplicando la primera derivada
def primera_derivada(imagen, sigma, borde):

	# Calculamos el tamanio
	tamanio = 3*sigma*2+1

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
	bordes_horizontales = cv2.normalize(bordes_horizontales, bordes_horizontales, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
	bordes_verticales = cv2.normalize(bordes_verticales, bordes_verticales, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

	representar_imagenes([imagen, bordes_horizontales,bordes_verticales],
		['Original alisada','1ª Bordes Horizontales','1ª Bordes Verticales'])


## APARTADO E ##

# Función de convolución con núcleo de segunda derivada
# Entrada:
#  - imagen: nombre del archivo al cual se le va a aplicar el filtro
#  - sigma: valor del sigma
#  - borde: borde que queremos aplicar a la imagen
# Salida: visualización de imágenes aplicando la segunda derivada
def segunda_derivada(imagen, sigma, borde):

	# Calculamos el tamanio
	tamanio = 3*sigma*2 + 1

	# Hacemos la segunda derivada del núcleo y nos devolverá una para X y otra para Y
	mascara = cv2.getDerivKernels(dx=2, dy=2, ksize=tamanio)
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
	bordes_horizontales = cv2.normalize(bordes_horizontales, bordes_horizontales, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
	bordes_verticales = cv2.normalize(bordes_verticales, bordes_verticales, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

	representar_imagenes([imagen, bordes_horizontales,bordes_verticales],
		['Original alisada','2ª Bordes Horizontales','2ª Bordes Verticales'])


## APARTADO F ##

# Función de convolución con núcleo Laplaciana-Gaussiana
# Entrada:
#  - imagen: nombre del archivo al cual se le va a aplicar el filtro
#  - sigma: valor del sigma
#  - borde: borde que queremos aplicar a la imagen
# Salida: visualización de imágenes aplicando la laplaciana
def laplaciana(imagen, sigma, borde):

	# Calculamos el tamanio
	tamanio = 3*sigma*2 + 1

	# Hacemos la segunda derivada del núcleo y nos devolverá una para X y otra para Y
	mascara = cv2.getDerivKernels(dx=2, dy=2, ksize=tamanio)
	# Sumamos ambas derivadas
	laplaciana = mascara[0] + mascara[1]

	# Normalizamos la imagen
	imagen = cv2.normalize(imagen, imagen, 0, 255, cv2.NORM_MINMAX)

	# Hacemos convolución
	laplace = cv2.filter2D(imagen, -1, kernel = laplaciana, borderType=borde)

	# representar_imagenes([imagen, laplace], ['imagen_original','laplaciana'])
	return laplace.astype(np.uint8)


## APARTADO G ##

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
		nueva_imagen = np.empty((np.shape(imagen)[0],math.ceil(np.shape(imagen)[1]*1.5),
			np.shape(imagen)[2]))

		# Meto la primera imagen a mano (estará a la izquierda)
		nueva_imagen[0:np.shape(gaussianPyramid[0])[0],
		0:np.shape(gaussianPyramid[0])[1], 0:3] = gaussianPyramid[0]
		# Meto la segunda imagen a mano (estará a la derecha)
		nueva_imagen[0:np.shape(gaussianPyramid[1])[0],
		np.shape(gaussianPyramid[0])[1]:math.ceil(np.shape(gaussianPyramid[1])[1]+np.shape(gaussianPyramid[0])[1]),
		0:3] = gaussianPyramid[1]

		# Recorro los niveles que me quedan
		for i in range(2, levels):

			# Calculo la altura de cada imagen, para añadirla a continuación de la segunda imagen
			# Por eso voy sumando las alturas, para que así vaya una detrás de otra
			altura = 0
			for j in range (1, i):
				altura = altura + np.shape(gaussianPyramid[j])[0]

			# Ya tengo mi imagen con la pirámide en color (los voy metiendo en mi imagen ventana)
			nueva_imagen[altura:(np.shape(gaussianPyramid[i])[0]+altura),
			np.shape(gaussianPyramid[0])[1]:math.ceil(np.shape(gaussianPyramid[i])[1]+np.shape(gaussianPyramid[0])[1]),
			0:3] = gaussianPyramid[i]

	# Compruebo si la imagen es en blanco y negro
	else:

		# Creo una imagen, con el tamaño de todas las imágenes a meter
		nueva_imagen = np.empty((np.shape(imagen)[0],math.ceil(np.shape(imagen)[1]*1.5)))

		# Meto la primera imagen a mano (estará a la izquierda)
		nueva_imagen[0:np.shape(gaussianPyramid[0])[0],
		0:np.shape(gaussianPyramid[0])[1]] = gaussianPyramid[0]
		# Meto la segunda imagen a mano (estará a la derecha)
		nueva_imagen[0:np.shape(gaussianPyramid[1])[0],
		np.shape(gaussianPyramid[0])[1]:math.ceil(np.shape(gaussianPyramid[1])[1]+np.shape(gaussianPyramid[0])[1])]= gaussianPyramid[1]

		# Recorro los niveles que me quedan
		for i in range(2, levels):

			# Calculo la altura de cada imagen, para añadirla a continuación de la segunda imagen
			# Por eso voy sumando las alturas, para que así vaya una detrás de otra
			altura = 0
			for j in range (1, i):
				altura = altura + np.shape(gaussianPyramid[j])[0]

			# Ya tengo mi imagen con la pirámide en color (los voy metiendo en mi imagen ventana)
			nueva_imagen[altura:(np.shape(gaussianPyramid[i])[0]+altura),
			np.shape(gaussianPyramid[0])[1]:math.ceil(np.shape(gaussianPyramid[i])[1]+np.shape(gaussianPyramid[0])[1])] = gaussianPyramid[i]

	# Devuelvo la pirámide
	return nueva_imagen.astype(np.uint8)


## APARTADO H ##

# Función que genera una representanción en pirámide Laplaciana
# Entrada:
#  - imagen: nombre del archivo al cual se le va a aplicar el filtro
#  - levels: niveles de la pirámide
#  - borde: borde a usar
# Salida: visualización de una imagen en pirámide Laplaciana
def piramide_laplaciana(imagen, levels):

	# Aplicar mi función laplaciana a la imagen original
	imagen = alisar_imagen(imagen, 1, borde=cv2.BORDER_REFLECT)
	imagenL = laplaciana(imagen, 1, cv2.BORDER_REFLECT)
	# imagen = alisar_imagen(imagen, 3, borde=cv2.BORDER_REPLICATE)
	# imagenL = laplaciana(imagen, 1, cv2.BORDER_REPLICATE)

	# Genero una copia de la imagem de entrada
	laplacianPyramid = [imagenL]

	# Recorro los niveles
	for i in range(1, levels):

		# Alisado y reescala
		img = cv2.pyrDown(laplacianPyramid[i - 1])
		# Laplaciana
		# LA IMAGEN YA TIENE HECHA LA LAPLACIANA
		# img = laplaciana(img, 1, cv2.BORDER_REFLECT)  ###COGRREGIDO
		# Creo una lista con las imagenes laplacianas
		laplacianPyramid.append(img)

	# Compruebo si la imagen es en color
	if (len(np.shape(imagen))) == 3:

		# Creo una matriz imagen, con el tamaño de todas las imágenes a meter
		nueva_imagen = np.empty((np.shape(imagen)[0], math.ceil(np.shape(imagen)[1]*1.5),
			np.shape(imagen)[2]))

		# Meto la primera imagen a mano (estará a la izquierda)
		nueva_imagen[0:np.shape(laplacianPyramid[0])[0],
		0:np.shape(laplacianPyramid[0])[1], 0:3] = laplacianPyramid[0]
		# Meto la segunda imagen a mano (estará a la derecha)
		nueva_imagen[0:np.shape(laplacianPyramid[1])[0],
		np.shape(laplacianPyramid[0])[1]:math.ceil(np.shape(laplacianPyramid[1])[1]+np.shape(laplacianPyramid[0])[1]),
		0:3] = laplacianPyramid[1]

		# Recorro los niveles que me quedan
		for i in range(2, levels):

			# Calculo la altura de cada imagen, para añadirla a continuación de la segunda imagen
			# Por eso voy sumando las alturas, para que así vaya una detrás de otra
			altura = 0
			for j in range (1, i):
				altura = altura + np.shape(laplacianPyramid[j])[0]

			# Ya tengo mi imagen con la pirámide en color
			nueva_imagen[altura:(np.shape(laplacianPyramid[i])[0]+altura),
			np.shape(laplacianPyramid[0])[1]:math.ceil(np.shape(laplacianPyramid[i])[1]+np.shape(laplacianPyramid[0])[1]),
			0:3] = laplacianPyramid[i]

	# Compruebo si la imagen es en blanco y negro
	else:
		# Creo una matriz imagen, con el tamaño de todas las imágenes a meter
		nueva_imagen = np.empty((np.shape(imagen)[0],
			math.ceil(np.shape(imagen)[1]*1.5)))

		# Meto la primera imagen a mano (estará a la izquierda)
		nueva_imagen[0:np.shape(laplacianPyramid[0])[0],
		0:np.shape(laplacianPyramid[0])[1]] = laplacianPyramid[0]
		# Meto la segunda imagen a mano (estará a la derecha)
		nueva_imagen[0:np.shape(laplacianPyramid[1])[0],
		np.shape(laplacianPyramid[0])[1]:math.ceil(np.shape(laplacianPyramid[1])[1]+np.shape(laplacianPyramid[0])[1])] = laplacianPyramid[1]

		# Recorro los niveles que me quedan
		for i in range(2, levels):

			# Calculo la altura de cada imagen, para añadirla a continuación de la segunda imagen
			# Por eso voy sumando las alturas, para que así vaya una detrás de otra
			altura = 0
			for j in range (1, i):
				altura = altura + np.shape(laplacianPyramid[j])[0]

			# Ya tengo mi imagen con la pirámide en color
			nueva_imagen[altura:(np.shape(laplacianPyramid[i])[0]+altura),
			np.shape(laplacianPyramid[0])[1]:math.ceil(np.shape(laplacianPyramid[i])[1]+np.shape(laplacianPyramid[0])[1])] = laplacianPyramid[i]

	return nueva_imagen.astype(np.uint8)



# ------------------------------------------------------------------------------------------------------- #
# EJERCICIO 2
# ------------------------------------------------------------------------------------------------------- #

# Función que genera una imagen en baja frecuencia
# Entrada:
#  - imagen: nombre del archivo al cual se le va a aplicar
#  - sigma: valor sigma a pasar
# Salida: matriz de la imagen con baja frecuencia
def genera_baja_frecuencia(imagen, sigma):

	# Calculo el tamanio de la mascara
	tamanio = 6*sigma+1
	# Filtro la imagen con un filtro Gaussiano para alisarla
	mascara = cv2.getGaussianKernel(ksize = tamanio, sigma = sigma)
	# filtrada = cv2.GaussianBlur(imagen, (tamanio,tamanio), sigma)
	filtrada = convolucion_separable(imagen, cv2.BORDER_REFLECT, mascara, mascara)

	# return filtrada.astype(np.uint8)
	return filtrada

# Función que genera una imagen en alta frecuencia
# Entrada:
#  - imagen: nombre del archivo al cual se le va a aplicar
#  - sigma: valor sigma a pasar
# Salida: matriz de la imagen con alta frecuencia
def genera_alta_frecuencia(imagen, sigma):
	# Nos quedamos con los detalles de la imagen al restarle a esta su alisada
	nueva_imagen = imagen - genera_baja_frecuencia(imagen, sigma)

	return nueva_imagen

# Función que genera una imagen hibrida
# Entrada:
#  - imagenA: imagen a aplicar el la alta frecuencia
#  - imagenB: imagen a aplicar el la baja frecuencia
#  - sigmaA: valor de sigma de la alta frecuencia
#  - sigmaB: valor de sigma de la baja frecuencia
# Salida1: imagen(matriz) de la híbrida
# Salida2: imagen ventana con las tres imágenes (alta, baja, hibrida)
def genera_hibrida_gris(imagenA, imagenB, sigmaA, sigmaB):

	# Genero las imagenes de alta y baja frecuencia
	img_alta = genera_alta_frecuencia(imagenA, sigmaA);
	img_baja = genera_baja_frecuencia(imagenB, sigmaB);

	# Las sumo
	hibrida = img_baja + img_alta

	# Normalizo la hibdrida por si hay valores negativos o superiores a 255
	hibrida = cv2.normalize(hibrida,hibrida,0,255, cv2.NORM_MINMAX)

	imagen_compuesta = np.empty((np.shape(img_alta)[0],
						np.shape(img_alta)[1]*3))
	imagen_compuesta[0:np.shape(img_alta)[0],
					0:np.shape(img_alta)[1]] = img_alta
	imagen_compuesta[0:np.shape(img_baja)[0],
					np.shape(img_alta)[1]:np.shape(img_alta)[1]+np.shape(img_baja)[1]] = img_baja
	imagen_compuesta[0:np.shape(hibrida)[0],
					np.shape(img_alta)[1]+np.shape(img_baja)[1]:np.shape(img_alta)[1]+np.shape(img_baja)[1]+np.shape(hibrida)[1],
					] = hibrida

	#return hibrida.astype(np.uint8)
	return imagen_compuesta.astype(np.uint8)
	#return (hibrida.astype(np.uint8),imagen_compuesta.astype(np.uint8))



# ------------------------------------------------------------------------------------------------------- #
# BONUS 1
# ------------------------------------------------------------------------------------------------------- #

# Funcion gaussiana para la mascara
# Implementamos la función que viene definida en el apartado
def funcion(x, sigma):
	k = np.exp(-0.5 * ((x**2)/(sigma**2)))
	return k

# Función que realiza el mismo proceso que getGaussianKernel
def calcular_mascara_gaussiana(sigma):

	# Debemos tomar 3*sigma por cada lado, siendo el tamaño total de la máscara: 6*sigma + 1
	tam = 3*sigma
	kernel = np.arange(-math.ceil(tam), math.ceil(tam+1))

	# Aplicamos la gaussiana a la máscara
	kernel = np.array([funcion(i, sigma) for i in kernel])

	# Y la devolvemos normalizada (algo normalizado su su suma da 1)
	normalizada = kernel/(sum(kernel))
	# print(np.sum(normalizada))

	return normalizada



# ------------------------------------------------------------------------------------------------------- #
# BONUS 2
# ------------------------------------------------------------------------------------------------------- #

# Función que calcula la convolución de un vector señal 1D con un vector máscara
def convolucion_1D(mascara_gaussiana, vector_imagen, borde):

	# Estoy copiando la estructura de la imagen (copiando la matriz sin los valores)
	nueva_imagen = np.empty(np.shape(vector_imagen))

	# Por cada canal paso la mascara (asi que me guardo el canal de la imagen) si es color es 3 y si es blanco y negro es 1
	# Obtengo el número de canales
	canales = len(np.shape(vector_imagen))

	# Cantidad de elementos que hay a la izquierda o a la derecha del elemento central de la máscara
	len_mascara_gaussiana = len(mascara_gaussiana)
	cantidad = (len_mascara_gaussiana - 1) // 2;

	# Longitud del vector
	len_vector_imagen = len(vector_imagen)

	# Si la imagen es en gris
	if canales == 1:

		# Borde reflejo
		if (borde == 0):
			# Ahora tenemos que distinguir la posibilidad de que de si la máscara se pone
			# a la derecha, al centro o a la izquierda

			# Posibilidad a la izquierda --> índice va de 0 a cantidad
			# Muevo la mascara sobre el array de la imagen (la parte izquierda de la mascara sobresale)
			for j in range(0, cantidad): # tengo una parte de la mascara fuera
				valor = 0

				# Tengo mi mascara e itero sobre los elementos de la mascara para multiplicarlos con los del array
				for k in range(0, len_mascara_gaussiana):

					# i es la dimensión, es el canal -> abs((0-1+0))
					valor = valor + vector_imagen[abs((j-cantidad+k)+1)] * mascara_gaussiana[k]

				# hemos cogido la mascara a solo un valor
				nueva_imagen[j] = valor

			# La mascara esta en el centro (ningun valor de la mascara sobresale)
			for j in range(cantidad, len_vector_imagen): # tengo la mascara dentro
				valor = 0

				# Tengo mi mascara e itero sobre los elementos de la máscara para multiplicarlos con los del array
				for k in range(0, len_mascara_gaussiana):

					if (j-cantidad + k < len_vector_imagen):
						# i es la dimensión, es el canal abs((0-1+0))
						valor = valor + vector_imagen[j-cantidad+k] * mascara_gaussiana[k]
					else:
						# La mascara esta a la derecha (la parte derecha de la mascara sobresale)
						valor = valor + vector_imagen[(len_vector_imagen-((j-cantidad+k)-len_vector_imagen)-1)] * mascara_gaussiana[k]

				# hemos cogido la mascara a solo un valor
				nueva_imagen[j] = valor

	# Si la imagen es en color
	else:
		# Recorro los canales
		for i in range(canales+1):

			# Borde reflejo
			if (borde == 0):
				# Ahora distinguimos las tres posibilidades
				# de si se pone la mascara a la derecha, al centro o a la izquierda

				# Posibilidad a la izquierda --> indice va de 0 a cantidad
				# Muevo la mascara sobre el array de la imagen (la parte izquierda de la mascara sobresale)
				for j in range(0, cantidad): # tengo una parte de la mascara fuera
					valor = 0

					# Tengo mi mascara e itero sobre los elementos de la mascara para multiplicarlos con los del array
					for k in range(0, len_mascara_gaussiana):

						# i es la dimensión, es el canal abs((0-1+0))
						valor = valor + vector_imagen[abs((j-cantidad+k)+1),i] * mascara_gaussiana[k]

					# Hemos cogido la mascara a solo un valor
					nueva_imagen[j,i] = valor

				# La mascara esta en el centro (ningun valor de la mascara sobresale)
				for j in range(cantidad, len_vector_imagen): # tengo la mascara dentro
					valor = 0

					# Tengo mi mascara e itero sobre los elementos de la macara para multiplicarlos con los del array
					for k in range(0, len_mascara_gaussiana):

						if (j-cantidad + k < len_vector_imagen):
							# i es la dimensión es el canal abs((0-1+0))
							valor = valor + vector_imagen[j-cantidad+k,i] * mascara_gaussiana[k]
						else:
							# La mascara esta a la derecha (la parte derecha de la mascara sobresale)
							valor = valor + vector_imagen[(len_vector_imagen-((j-cantidad+k)-len_vector_imagen)-1),i] * mascara_gaussiana[k]

					# Hemos cogido la mascara a solo un valor
					nueva_imagen[j,i] = valor

	return nueva_imagen.astype(np.uint8)



# ------------------------------------------------------------------------------------------------------- #
# BONUS 3
# ------------------------------------------------------------------------------------------------------- #
# Hay que convolucionar la imagen con esa convolución1D
def convolucionar_imagen(mascara_x, mascara_y, imagen2D, borde):

	# Estoy copiando la estructura de la imagen (copiando la matriz sin los valores)
	nueva_imagen = np.empty(np.shape(imagen2D))
	
	# Si imagen en escala de grises B/N
	if len(np.shape(imagen2D)) == 2:

		# Por filas convolucionando
		# Recorro la imagen por filas para aplicarle el filtro Gaussiano 1D
		for i in range(np.shape(imagen2D)[0]):
			# Hago convolucion
			fila = convolucion_1D(mascara_x, imagen2D[i,:], borde)
			nueva_imagen[i,:] = fila

		# Transponemos la imagen
		# al transponer lo que cambiamos es filas por columnas
		nueva_imagen = nueva_imagen.transpose()

		# Ahora hacemos convolucion a las columnas
		# pero como hemos transpuesto serán las filas
		for i in range (np.shape(nueva_imagen)[0]):
			# Hago convolucion
			fila = convolucion_1D(mascara_y, nueva_imagen[i,:], borde)
			nueva_imagen[i,:] = fila

		# Vuelvo a la imagen original
		nueva_imagen = nueva_imagen.transpose()

	else:

		# Por filas convolucionando
		# Recorro la imagen por filas para aplicarle el filtro Gaussiano 1D
		for i in range(np.shape(imagen2D)[0]):
			# Hago convolucion
			fila = convolucion_1D(mascara_x, imagen2D[i,:,:], borde)
			nueva_imagen[i,:,:] = fila

		# Transponemos la imagen
		# al transponer lo que cambiamos es filas por columnas
		nueva_imagen = nueva_imagen.transpose(1,0,2)

		# Ahora hacemos convolucion a las columnas
		# pero como hemos transpuesto serán las filas
		for i in range (np.shape(nueva_imagen)[0]):
			# Hago convolucion
			fila = convolucion_1D(mascara_y, nueva_imagen[i,:,:], borde)
			nueva_imagen[i,:,:] = fila

		# Vuelvo a la imagen original
		nueva_imagen = nueva_imagen.transpose(1,0,2)

	return nueva_imagen.astype(np.uint8)


# ------------------------------------------------------------------------------------------------------- #
# BONUS 3
# ------------------------------------------------------------------------------------------------------- #
def my_pyrDown(imagen):

	# Aplicamos un alisado (convolucionar_imagen)
	# Y nos quedamos con las  columnas y filas pares para reescalar

	# Hacemos convolución
	sigma = 1
	m_x = calcular_mascara_gaussiana(sigma)
	m_y = calcular_mascara_gaussiana(sigma)
	alisado = convolucionar_imagen(m_x, m_y, imagen, 0)

	# Nos creamos una nueva imagen para guardar
	nueva_imagen = alisado[range(0,alisado.shape[0],2)]
	# y nos quedamos con las con las filas y columnas pares
	nueva_imagen = nueva_imagen[:,range(0,alisado.shape[1],2)]

	return nueva_imagen.astype(np.uint8)

# Función que genera una representanción en pirámide Gaussiana
def piramide_gaussiana_bonus(imagen, levels, borde):

	# Generamos una copia de la imagen original
	gaussianPyramid = [imagen.copy()]

	for i in range(1, levels):

		# Aplicamos my_pyrDown
		img = my_pyrDown(gaussianPyramid[i - 1])
		gaussianPyramid.append(img)

	# Si la imagen es en color
	if (len(np.shape(imagen))) == 3:

		# Creo una imagen, con el tamaño de todas las imágenes a meter
		nueva_imagen = np.empty((np.shape(imagen)[0],math.ceil(np.shape(imagen)[1]*1.5),
			np.shape(imagen)[2]))

		# La primera la metemos a mano
		nueva_imagen[0:np.shape(gaussianPyramid[0])[0], 0:np.shape(gaussianPyramid[0])[1], 0:3] = gaussianPyramid[0]
		# La segunda la metemos a mano
		nueva_imagen[0:np.shape(gaussianPyramid[1])[0],
		np.shape(gaussianPyramid[0])[1]:math.ceil(np.shape(gaussianPyramid[1])[1]+np.shape(gaussianPyramid[0])[1]),
		0:3] = gaussianPyramid[1]

		# Recorremos los niveles que quedan
		for i in range(2, levels):

			# Calculamos la altura para colocar las imagenes una detrás de otra
			altura = 0
			for j in range (1, i):
				altura = altura + np.shape(gaussianPyramid[j])[0]

			# Ya tenemos nuestra imagen ventana con la pirámide
			nueva_imagen[altura:(np.shape(gaussianPyramid[i])[0]+altura),
			np.shape(gaussianPyramidB[0])[1]:math.ceil(np.shape(gaussianPyramid[i])[1]+np.shape(gaussianPyramid[0])[1]),
			0:3] = gaussianPyramid[i]

	# Compruebo si la imagen es en blanco y negro
	else:

		# Creo una imagen, con el tamaño de todas las imágenes a meter
		nueva_imagen = np.empty((np.shape(imagen)[0],math.ceil(np.shape(imagen)[1]*1.5)))

		# Meto la primera imagen a mano (estará a la izquierda)
		nueva_imagen[0:np.shape(gaussianPyramid[0])[0],
		0:np.shape(gaussianPyramid[0])[1]] = gaussianPyramid[0]
		# Meto la segunda imagen a mano (estará a la derecha)
		nueva_imagen[0:np.shape(gaussianPyramid[1])[0],
		np.shape(gaussianPyramid[0])[1]:math.ceil(np.shape(gaussianPyramid[1])[1]+np.shape(gaussianPyramid[0])[1])]= gaussianPyramid[1]

		# Recorro los niveles que me quedan
		for i in range(2, levels):

			# Calculo la altura de cada imagen, para añadirla a continuación de la segunda imagen
			# Por eso voy sumando las alturas, para que así vaya una detrás de otra
			altura = 0
			for j in range (1, i):
				altura = altura + np.shape(gaussianPyramid[j])[0]

			# Ya tengo mi imagen con la pirámide en color
			nueva_imagen[altura:(np.shape(gaussianPyramid[i])[0]+altura),
			np.shape(gaussianPyramid[0])[1]:math.ceil(np.shape(gaussianPyramid[i])[1]+np.shape(gaussianPyramid[0])[1])] = gaussianPyramid[i]

	return nueva_imagen.astype(np.uint8)



# ------------------------------------------------------------------------------------------------------- #
# BONUS 4
# ------------------------------------------------------------------------------------------------------- #
# Función que genera una imagen en baja frecuencia
# Entrada:
#  - imagen: nombre del archivo al cual se le va a aplicar
#  - sigma: valor sigma a pasar
# Salida: matriz de la imagen con baja frecuencia
def genera_baja_frecuencia_bonus(imagen, sigma):

	# Filtro la imagen con un filtro Gaussiano para alisarla
	mascara = calcular_mascara_gaussiana(sigma)
	# filtrada = cv2.GaussianBlur(imagen, (tamanio,tamanio), sigma)
	filtrada = convolucionar_imagen(mascara, mascara, imagen, 0)
	# return filtrada.astype(np.uint8)
	return filtrada

# Función que genera una imagen en alta frecuencia
# Entrada:
#  - imagen: nombre del archivo al cual se le va a aplicar
#  - sigma: valor sigma a pasar
# Salida: matriz de la imagen con alta frecuencia
def genera_alta_frecuencia_bonus(imagen, sigma):
	# Nos quedamos con los detalles de la imagen al restarle a esta su alisada
	nueva_imagen = imagen - genera_baja_frecuencia_bonus(imagen, sigma)

	return nueva_imagen

# Función que genera una imagen hibrida
# Entrada:
#  - imagenA: imagen a aplicar el la alta frecuencia
#  - imagenB: imagen a aplicar el la baja frecuencia
#  - sigmaA: valor de sigma de la alta frecuencia
#  - sigmaB: valor de sigma de la baja frecuencia
# Salida1: imagen(matriz) de la híbrida
# Salida2: imagen ventana con las tres imágenes (alta, baja, hibrida)

#def genera_hibrida(imagenA, imagenB, sigmaA, sigmaB, tamanioA, tamanioB):
def genera_hibrida_bonus(imagenA, imagenB, sigmaA, sigmaB):

	tamanioA = 6*sigmaA+1
	tamanioB = 6*sigmaB+1
	# Estoy copiando la estructura de la imagen (copiando la matriz sin los valores)
	# img_alta = np.empty(np.shape(imagenA))
	# img_baja = np.empty(np.shape(imagenB))

	img_alta = genera_alta_frecuencia_bonus(imagenA, sigmaA);
	img_baja = genera_baja_frecuencia_bonus(imagenB, sigmaB);
	hibrida = img_baja + img_alta

	# Vamos a crear una nueva imagen componiendo 3 imagenes (alta, baja e hibrida)
	# Se supone que ambas imágenes (imagenA e imagenB) miden lo mismo
	if len(np.shape(imagenA)) == 3:
		imagen_compuesta = np.empty((np.shape(img_alta)[0],
							 np.shape(img_alta)[1]*3,
							 np.shape(img_alta)[2]))
		imagen_compuesta[0:np.shape(img_alta)[0],
						 0:np.shape(img_alta)[1],
						 0:3] = img_alta
		imagen_compuesta[0:np.shape(img_baja)[0],
						 np.shape(img_alta)[1]:np.shape(img_alta)[1]+np.shape(img_baja)[1],
						 0:3] = img_baja
		imagen_compuesta[0:np.shape(hibrida)[0],
						 np.shape(img_alta)[1]+np.shape(img_baja)[1]:np.shape(img_alta)[1]+np.shape(img_baja)[1]+np.shape(hibrida)[1],
					 0:3] = hibrida
	else:
		imagen_compuesta = np.empty((np.shape(img_alta)[0],
							np.shape(img_alta)[1]*3))
		imagen_compuesta[0:np.shape(img_alta)[0],
						0:np.shape(img_alta)[1]] = img_alta
		imagen_compuesta[0:np.shape(img_baja)[0],
						np.shape(img_alta)[1]:np.shape(img_alta)[1]+np.shape(img_baja)[1]] = img_baja
		imagen_compuesta[0:np.shape(hibrida)[0],
						np.shape(img_alta)[1]+np.shape(img_baja)[1]:np.shape(img_alta)[1]+np.shape(img_baja)[1]+np.shape(hibrida)[1]] = hibrida

	return hibrida.astype(np.uint8)
	# return imagen_compuesta.astype(np.uint8)


if __name__ == '__main__':

	
	######################################################################################################
	# EJERCICIO 1
	######################################################################################################

	# EJERCICIO 1 A - Representación de imágenes
	img1 = leer_imagen('imagenes/cat.bmp', True)
	img2 = leer_imagen('imagenes/cat.bmp', False)
	representar_imagenes([img1,img2], ['Leemos en color','Transformamos a gris'])

	# cv2.waitKey()
	# cv2.destroyAllWindows()

	# EJERCICIO 1B - Convolución con máscara gaussiana
	# img1 = leer_imagen('imagenes/cat.bmp', True)
	img2 = alisar_imagen(img1, sigma=3, borde = cv2.BORDER_CONSTANT)
	img3 = alisar_imagen(img1, sigma=3, borde = cv2.BORDER_REFLECT)
	img4 = alisar_imagen(img1, sigma=9, borde = cv2.BORDER_CONSTANT)
	img5 = alisar_imagen(img1, sigma=9, borde = cv2.BORDER_REFLECT)
	representar_imagenes([img1,img2,img3,img4,img5],
		['Original','BORDER_CONSTANT, σ=3','BORDER_REFLECT, σ=3','BORDER_CONSTANT, σ=9','BORDER_REFLECT, σ=9'])

	# EJERCICIO 1C - Convolución con núcleo separable
	# Calculo la máscara para filas y para columnas (usando dos sigmas distintos)
	mask_fila1 = cv2.getGaussianKernel(ksize=13, sigma=2)
	mask_col1 = mask_fila1
	mask_fila2 = cv2.getGaussianKernel(ksize=43, sigma=7)
	mask_col2 = mask_fila2
	# img1 = leer_imagen('imagenes/cat.bmp', flag_color=True)
	img2 = convolucion_separable(img1, cv2.BORDER_REPLICATE, mask_fila1, mask_col1)
	img3 = convolucion_separable(img1, cv2.BORDER_CONSTANT, mask_fila1, mask_col1)
	img4 = convolucion_separable(img1, cv2.BORDER_REPLICATE, mask_fila2, mask_col2)
	img5 = convolucion_separable(img1, cv2.BORDER_CONSTANT, mask_fila2, mask_col2)
	representar_imagenes([img1,img2,img3,img4,img5],
		['Original','BORDER_REPLICATE σ=2','BORDER_CONSTANT σ=2','BORDER_REPLICATE σ=7','BORDER_CONSTANT σ=7'])
	# Para comparar con la función de opencv
	mask_fila1 = cv2.getGaussianKernel(ksize=19, sigma=3)
	mask_col1 = mask_fila1
	img2 = convolucion_separable(img1, cv2.BORDER_REPLICATE, mask_fila1, mask_col1)
	img6 = cv2.sepFilter2D(img1, -1, kernelX = mask_fila1,
		kernelY = mask_col1, borderType = cv2.BORDER_REPLICATE)
	img6 = img6.astype(np.uint8) # Cambiamos la codificación
	# Para comparar si las matrgices son iguales
	# print(np.all(img2 == img3))
	# print(img2 == img3)
	# Visuzalizamos las matrices de ambas
	# print(img2, "\n", img6)
	representar_imagenes([img1,img2,img6], ['Original','my_sepfilter2D','opencv_sepFilter2D'])
	# Realizamos un pequeño experimiento para ver que si el valor de sigma es más grande por
	# para las filas, obtendremos un mayor alisado por filas que por columnas
	mask_fila = cv2.getGaussianKernel(ksize=19, sigma=3)
	mask_col = cv2.getGaussianKernel(ksize=43, sigma=7)
	img4 = leer_imagen('imagenes/cruz.bmp', flag_color=False)
	img5 = convolucion_separable(img4, cv2.BORDER_REFLECT, mask_fila, mask_col)
	representar_imagenes([img4,img5], ['Original','Imagen Modificada'])

	# EJERCICIO 1D - Primera Derivada
	img = leer_imagen('imagenes/cat.bmp', flag_color=False)
	# 1. Aliso la imagen antes - Sigma 1 y borde reflejo
	img1 = alisar_imagen(img, sigma = 1, borde = cv2.BORDER_REFLECT)
	img2 = primera_derivada(img1, sigma = 1, borde = cv2.BORDER_REFLECT)
	# 2. Aliso la imagen antes - Sigma 2 y borde constante
	img1 = alisar_imagen(img, sigma = 2, borde = cv2.BORDER_CONSTANT)
	img2 = primera_derivada(img1, sigma = 1, borde = cv2.BORDER_CONSTANT)

	# EJERCICIO 1E - Segunda Derivada
	img = leer_imagen('imagenes/cat.bmp', flag_color=False)
	# 1. Aliso la imagen antes - Sigma 1 y borde reflejo
	img1 = alisar_imagen(img, sigma = 1, borde = cv2.BORDER_REFLECT)
	img2 = segunda_derivada(img1, sigma = 1, borde = cv2.BORDER_REFLECT)
	# 2. Aliso la imagen antes - Sigma 2 y borde constante
	img1 = alisar_imagen(img, sigma = 2, borde = cv2.BORDER_CONSTANT)
	img2 = segunda_derivada(img1, sigma = 1, borde = cv2.BORDER_CONSTANT)

	# EJERCICIO 1F - Laplaciana
	img = leer_imagen('imagenes/cat.bmp', flag_color=False)
	# 1. Aliso la imagen antes - Sigma 1 y borde reflejo
	img1 = alisar_imagen(img, sigma = 1, borde = cv2.BORDER_REFLECT)
	img2 = laplaciana(img1, sigma = 1, borde = cv2.BORDER_REFLECT)
	# 2. Aliso la imagen antes - Sigma 2 y borde constante
	img3 = alisar_imagen(img, sigma = 2, borde = cv2.BORDER_CONSTANT)
	img4 = laplaciana(img3, sigma = 1, borde = cv2.BORDER_CONSTANT)
	representar_imagenes([img,img2,img4], ['Original','Laplaciana σ=1','Laplaciana σ=2'])
	# Para comparar mi función laplaciana con la de OpenCV
	# OpenCV Laplacian
	imagen = leer_imagen('imagenes/cat.bmp', flag_color=False)
	img = alisar_imagen(imagen, 1, borde=cv2.BORDER_REFLECT)
	img = cv2.Laplacian(img, -1, ksize=1, borderType=cv2.BORDER_REFLECT)
	img = img.astype(np.uint8)
	# Mía Laplacian
	img1 = leer_imagen('imagenes/cat.bmp', flag_color=False)
	img1 = alisar_imagen(img1, 1, borde=cv2.BORDER_REFLECT)
	img2 = laplaciana(img1, sigma = 1, borde = cv2.BORDER_REFLECT)
	# print(np.array(img2))
	# print(np.array(img))
	representar_imagenes([img2,img], ['my_laplacian','opencv_laplacian'])

	# EJERCICIO 1G - Piramide Gaussiana
	img1 =  leer_imagen('imagenes/plane.bmp', flag_color=True)
	img2 = piramide_gaussiana(img1, 5, borde=cv2.BORDER_REFLECT)
	img21 = piramide_gaussiana(img1, 5, borde=cv2.BORDER_REPLICATE)
	# cv2.imwrite("imagenes/piramide_gaussianaC.bmp", img2)
	img3 =  leer_imagen('imagenes/plane.bmp', flag_color=False)
	img4 = piramide_gaussiana(img3, 5, borde=cv2.BORDER_REFLECT)
	img41 = piramide_gaussiana(img3, 5, borde=cv2.BORDER_REPLICATE)
	# cv2.imwrite("imagenes/piramide_gaussianaG.bmp", img4)
	representar_imagenes([img2,img21,img4,img41],
		['Piramide Color (Reflect)','Piramide Color (Replicate)','Piramide Gris (Reflect)','Piramide Color (Replicate)'])

	# EJERCICIO 1H - Piramide Laplaciana
	imgL2 = piramide_laplaciana(img1, 5)
	# cv2.imwrite('imagenes/plane2.bmp',imgL2)
	imgL4 = piramide_laplaciana(img3, 5)
	# cv2.imwrite('imagenes/plane1.bmp',imgL4)
	representar_imagenes([imgL2,imgL4],['PiramideL en Color','PiramideL en Gris'])


	######################################################################################################
	# EJERCICIO 2
	######################################################################################################

	# Pajaro y Avion
	imgA =  leer_imagen('imagenes/plane.bmp', flag_color=False)
	imgB =  leer_imagen('imagenes/bird.bmp', flag_color=False)
	hibrida1 = genera_hibrida_gris(imgB,imgA,10,1)
	# cv2.imwrite("imagenes/hibrida1.bmp", hibrida)

	# Pez y submarino
	imgA =  leer_imagen('imagenes/fish.bmp', flag_color=False)
	imgB =  leer_imagen('imagenes/submarine.bmp', flag_color=False)
	hibrida2 = genera_hibrida_gris(imgB,imgA,10,2)
	# cv2.imwrite("imagenes/hibrida2.bmp", hibrida)

	# Gato y perro
	imgA =  leer_imagen('imagenes/cat.bmp', flag_color=False)
	imgB =  leer_imagen('imagenes/dog.bmp', flag_color=False)
	hibrida3 = genera_hibrida_gris(imgB,imgA,10,2)
	# cv2.imwrite("imagenes/hibrida4.bmp", hibrida)

	representar_imagenes([ hibrida1, hibrida2, hibrida3], ["Pajaro-Avion", "Pez-Submarino","Gato-Perro"])

	# Experimento
	i = piramide_gaussiana(hibrida3, 5, borde=cv2.BORDER_REFLECT)
	# representar_imagenes([i], ["Piramide Hibrida Gaussiana"])
	#cv2.imwrite("imagenes/i.bmp",i)


	######################################################################################################
	# Bonus 1
	######################################################################################################
	# my_getGaussianKernel
	#print(calcular_mascara_gaussiana(1))
	# opencv_getGaussianKernel
	#print(cv2.getGaussianKernel(7,1))


	######################################################################################################
	# Bonus 2
	######################################################################################################
	# mascara = cv2.getGaussianKernel(ksize=7,sigma=1)
	mascara = calcular_mascara_gaussiana(1)
	img1 = leer_imagen('imagenes/cat.bmp', flag_color=True)
	img2 = convolucion_1D(mascara,img1[0,:],0)
	#print("imagen salida:", img2)
	#print("imagen entrada:", img1)
	#print("mascara:", img2)


	####################################################################################################s##
	# Bonus 3
	######################################################################################################
	#mascara = cv2.getGaussianKernel(ksize=7,sigma=1)
	mascara_x = calcular_mascara_gaussiana(3)
	mascara_y = calcular_mascara_gaussiana(3)
	img1 = leer_imagen('imagenes/cat.bmp', flag_color=False)
	img2 = convolucionar_imagen(mascara_x, mascara_y,img1,0)
	# cv2.imwrite("imagenes/cat4.bmp", img2)
	representar_imagenes([img1, img2], ['Original','Modificada'])
	# Para comparar con OpenCV
	mask_fila1 = cv2.getGaussianKernel(ksize=19, sigma=3)
	mask_col1 = mask_fila1
	img2 = convolucion_separable(img1, cv2.BORDER_REFLECT, mask_fila1, mask_col1)
	img6 = cv2.sepFilter2D(img1, -1, kernelX = mask_fila1,
		kernelY = mask_col1, borderType = cv2.BORDER_REFLECT)
	#print(img6)
	#print(img2)


	######################################################################################################
	# Bonus 3 Segundo
	######################################################################################################
	# imgB =  leer_imagen('imagenes/bicycle.bmp', flag_color=True)
	# Pajaro y Avion
	imgA =  leer_imagen('imagenes/plane.bmp', flag_color=False)
	imgB =  leer_imagen('imagenes/bird.bmp', flag_color=False)
	hib1 = genera_hibrida_bonus(imgB,imgA,10,1)
	# cv2.imwrite("imagenes/hibrida1.bmp", hibrida)

	# Pez y submarino
	imgA =  leer_imagen('imagenes/fish.bmp', flag_color=False)
	imgB =  leer_imagen('imagenes/submarine.bmp', flag_color=False)
	hib2 = genera_hibrida_bonus(imgB,imgA,10,2)
	# cv2.imwrite("imagenes/hibrida2.bmp", hibrida)

	# Gato y perro
	imgA =  leer_imagen('imagenes/cat.bmp', flag_color=False)
	imgB =  leer_imagen('imagenes/dog.bmp', flag_color=False)
	hib3 = genera_hibrida_bonus(imgB,imgA,10,2)
	# cv2.imwrite("imagenes/hibrida4.bmp", hibrida)

	h1 = piramide_gaussiana_bonus(hib1,7,0)
	h2 = piramide_gaussiana_bonus(hib2,7,0)
	h3 = piramide_gaussiana_bonus(hib3,7,0)
	representar_imagenes([h1, h2, h3],['Hibrida 1', 'Hibrida 2', 'Hibrida3'] )
	#cv2.imwrite("imagenes/p.bmp",img1)


	######################################################################################################
	# Bonus 4
	######################################################################################################

	# Pajaro y Avion
	imgA =  leer_imagen('imagenes/plane.bmp', flag_color=True)
	imgB =  leer_imagen('imagenes/bird.bmp', flag_color=True)
	hibrida1 = genera_hibrida_bonus(imgB,imgA,10,1)
	# cv2.imwrite("imagenes/hibrida1.bmp", hibrida)

	# Pez y submarino
	imgA =  leer_imagen('imagenes/fish.bmp', flag_color=True)
	imgB =  leer_imagen('imagenes/submarine.bmp', flag_color=True)
	hibrida2 = genera_hibrida_bonus(imgB,imgA,10,2)
	# cv2.imwrite("imagenes/hibrida2.bmp", hibrida)

	# Gato y perro
	imgA =  leer_imagen('imagenes/cat.bmp', flag_color=True)
	imgB =  leer_imagen('imagenes/dog.bmp', flag_color=True)
	hibrida3 = genera_hibrida_bonus(imgB,imgA,10,2)
	# cv2.imwrite("imagenes/hibrida4.bmp", hibrida)

	representar_imagenes([ hibrida1, hibrida2, hibrida3], ["Pajaro-Avion", "Pez-Submarino","Gato-Perro"])
