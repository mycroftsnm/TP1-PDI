# Trabajo Práctico 1 - Procesamiento de Imágenes
Armas, Soda y Ferreira da Cámara

Este repositorio contiene los scripts para el Trabajo Práctico 1, dividido en dos problemas: ecualización local de histograma y validación automática de formularios.

## 1. Descripción de los Problemas

### Problema 1: Ecualización Local de Histograma
`Problema1-Ecualizacion_local_de_histograma.py` aplica la técnica de Ecualización de Histograma Adaptativa Limitada por Contraste (CLAHE) de OpenCV en la imagen `Imagen_con_detalles_escondidos.tif`. El script prueba varias dimensiones de ventana (10x10, 12x12, 16x16, 21x21) y muestra los resultados normalizados.

### Problema 2: Validación de Formularios
`Problema2-Validacion_de_formulario.py` procesa un conjunto de imágenes de formularios (`formulario_0N.png`). Analiza la estructura de la tabla, segmenta cada celda y aplica reglas de validación contando letras y palabras.

Genera tres salidas:
1.  Resultados de validación (`OK`/`MAL`) por campo en la consola.
2.  Un archivo `resultados.csv` con el resumen de todos los formularios.
3.  Una imagen que agrupa los campos "Nombre y Apellido" de los formularios correctos e incorrectos.

## 2. Requisitos
Python 3 y las siguientes bibliotecas:

* `opencv-python` (cv2)
* `numpy`
* `matplotlib`
* `pandas`