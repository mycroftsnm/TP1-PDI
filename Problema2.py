import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def contar_letras(imagen):
    """
    Recibe una imagen binaria (blanco sobre negro).
    Realiza dilatación moderada para obtener mejor detección de las letras.
    Devuelve el número de letras (contornos).
    """
    kernel = np.ones((3, 3), np.uint8)
    imagen_abierta = cv2.morphologyEx(imagen, cv2.MORPH_OPEN, kernel)

    contornos, _ = cv2.findContours(imagen_abierta, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return len(contornos)


def contar_palabras(imagen):
    """
    Cuenta palabras en una imagen binaria (blanco sobre negro).
    Usa dilatación morfológica para unir letras y detecta contornos separados.
    Retorna el número de palabras encontradas.
    """
    _, thresh = cv2.threshold(imagen, 0, 255, cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Retorna el número de quiebres de palabra detectados
    return len(contours)


def procesar_imagen_formulario(imagen):
    '''
    Recibe la imagen de un formulario y devuelve
    la imagen ajustada a los bordes de la tabla
    y binarizada.
    '''
    _, imagen = cv2.threshold(imagen, 220, 255, cv2.THRESH_BINARY_INV)
    coordenadas_y, coordenadas_x = np.where(imagen > 0)
    mx, my, Mx, My = np.min(coordenadas_x), np.min(coordenadas_y), np.max(coordenadas_x), np.max(coordenadas_y)

    return imagen[my:My+1, mx:Mx+1]

def obtener_celdas(imagen):
    celdas = []
    
    largo, ancho = imagen.shape
    suma_filas = np.sum(imagen, axis=1)
    filas = np.where(suma_filas == 255 * ancho)[0]

    for i in range(1,len(filas)-1,2):
        fila = []
        my = filas[i] + 1
        My = filas[i+1]
        suma_cols = np.sum(imagen[my:My,:], axis=0)
        cols = np.where(suma_cols == 255 * (My-my))[0]
        for j in range(1,len(cols)-1,2):
            mx = cols[j] + 1
            Mx = cols[j+1]
            fila.append(imagen[my:My,mx:Mx])
        celdas.append(fila)
    return celdas

def obtener_resultados(celdas):
    rtas = {}
    correcto = True

    # Nombre y apellido (fila 1, columna 1)
    if contar_palabras(celdas[1][1]) >= 2 and contar_letras(celdas[1][1]) <= 25:
        rtas["Nombre y Apellido"] = 'OK'
    else:
        rtas["Nombre y Apellido"] = 'MAL'
        correcto = False

    # Edad (fila 2, columna 1)
    if contar_palabras(celdas[2][1]) == 1 and contar_letras(celdas[2][1]) <= 3:
        rtas["Edad"] = 'OK'
    else:
        rtas["Edad"] = 'MAL'
        correcto = False

    # Mail (fila 3, columna 1)
    if contar_palabras(celdas[3][1]) == 1 and contar_letras(celdas[3][1]) <= 25:
        rtas["Mail"] = 'OK'
    else:
        rtas["Mail"] = 'MAL'
        correcto = False

    # Legajo (fila 4, columna 1)
    if contar_palabras(celdas[4][1]) == 1 and contar_letras(celdas[4][1]) <= 8:
        rtas["Legajo"] = 'OK'
    else:
        rtas["Legajo"] = 'MAL'
        correcto = False
    
    # Pregunta 1 (fila 6, columnas 1 y 2)
    if (contar_letras(celdas[6][1]) + contar_letras(celdas[6][2])) == 1:
        rtas["Pregunta 1"] = 'OK'
    else:
        rtas["Pregunta 1"] = 'MAL'
        correcto = False

    # Pregunta 2 (fila 7, columnas 1 y 2)
    if (contar_letras(celdas[7][1]) + contar_letras(celdas[7][2])) == 1:
        rtas["Pregunta 2"] = 'OK'
    else:
        rtas["Pregunta 2"] = 'MAL'
        correcto = False

    # Pregunta 3 (fila 8, columnas 1 y 2)
    if (contar_letras(celdas[8][1]) + contar_letras(celdas[8][2])) == 1:
        rtas["Pregunta 3"] = 'OK'
    else:
        rtas["Pregunta 3"] = 'MAL'
        correcto = False

    # Comentarios (fila 9, columna 1)
    if contar_palabras(celdas[9][1]) > 1 and contar_letras(celdas[9][1]) <= 25:
        rtas["Comentarios"] = 'OK'
    else:
        rtas["Comentarios"] = 'MAL'
        correcto = False

    return rtas, correcto
    
def generar_imagen_resultados(correctos, incorrectos):
    img_correctos = cv2.vconcat(correctos)
    img_incorrectos = cv2.vconcat(incorrectos)

    # Invertir de nuevo para volver al negro sobre blanco original
    img_correctos = cv2.bitwise_not(img_correctos)
    img_incorrectos = cv2.bitwise_not(img_incorrectos)

    alto_correctos, ancho_correctos = img_correctos.shape[:2]
    alto_incorrectos, ancho_incorrectos = img_incorrectos.shape[:2]

    color_fondo = 255
    imagen_correctos_con_texto = np.ones((alto_correctos + 50, ancho_correctos), dtype=np.uint8) * color_fondo
    imagen_incorrectos_con_texto = np.ones((alto_incorrectos + 50, ancho_incorrectos), dtype=np.uint8) * color_fondo

    font = cv2.FONT_HERSHEY_SIMPLEX
    texto_correctos = "Correctos"
    tamaño_fuente = 1
    color_texto = 0
    grosor = 2  
    (tamaño_texto, _) = cv2.getTextSize(texto_correctos, font, tamaño_fuente, grosor)
    pos_x = (ancho_correctos - tamaño_texto[0]) // 2
    pos_y = 30  
    cv2.putText(imagen_correctos_con_texto, texto_correctos, (pos_x, pos_y), font, tamaño_fuente, color_texto, grosor)
    cv2.line(imagen_correctos_con_texto, (0, 45), (ancho_correctos, 45), color_texto, 2)

    texto_incorrectos = "Incorrectos"
    (tamaño_texto, _) = cv2.getTextSize(texto_incorrectos, font, tamaño_fuente, grosor)
    pos_x_fallo = (ancho_incorrectos - tamaño_texto[0]) // 2
    pos_y_fallo = 30  
    cv2.putText(imagen_incorrectos_con_texto, texto_incorrectos, (pos_x_fallo, pos_y_fallo), font, tamaño_fuente, color_texto, grosor)
    cv2.line(imagen_incorrectos_con_texto, (0, 45), (ancho_incorrectos, 45), color_texto, 2)

    imagen_correctos_con_texto[50:, :] = img_correctos
    imagen_incorrectos_con_texto[50:, :] = img_incorrectos

    imagen_final = np.vstack([imagen_correctos_con_texto, imagen_incorrectos_con_texto])

    plt.figure()
    plt.imshow(imagen_final, cmap='gray')
    plt.axis('off')
    plt.show()

def identificar_formulario(crop_celda):
    '''
    Recibe el crop de la celda del título 
    del formulario y devuelve que tipo de formulario es.
    Si no puede identificarlo devuelve 'X'
    '''

    # Precomputados
    AREA_LETRA_A = 134
    AREA_LETRA_B = 152

    # Si no tiene exactamente 11 letras no esta pudiendo
    # identificar correctamente el texto, devuelve 'X'.
    # 'FORMULARIO' 10 letras + 1 letra 'A', 'B' o 'C' que identifica el tipo 
    if contar_letras(crop_celda) != 11:
        return 'X'
    
    # Obtenemos las componentes conectadas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        crop_celda, 8, cv2.CV_32S
    )
    # índice de la componente más a la derecha (última letra)
    idx = np.argmax(stats[:, cv2.CC_STAT_LEFT]) 

    area_letra = stats[idx][cv2.CC_STAT_AREA]

    if abs(area_letra - AREA_LETRA_A) < 3:
        return 'A'
    elif abs(area_letra - AREA_LETRA_B) < 3:
        return 'B'
    else:
        return 'C'

if __name__ == "__main__":

    correctos = [] # Lista de crops del campo Nombre de formularios correctos
    incorrectos = [] # Lista de crops del campo Nombre de formularios incorrectos
    lista_resultados = [] # Lista con todos los diccionarios de resultados de cada formulario

    for i in range(5):
        img = cv2.imread(f'formulario_0{i+1}.png', cv2.IMREAD_GRAYSCALE)

        img = procesar_imagen_formulario(img)
        celdas = obtener_celdas(img)

        print(f"\nFormulario {i+1} tipo {identificar_formulario(celdas[0][0])}:")
        
        resultados, es_correcto = obtener_resultados(celdas)
        for campo, resultado in resultados.items():
            print(f"{campo}: {resultado}")
        
        resultados['ID'] = i+1
        lista_resultados.append(resultados)

        if es_correcto:
            correctos.append(celdas[1][1])
        else:
            incorrectos.append(celdas[1][1])

    generar_imagen_resultados(correctos, incorrectos)

    df_resultados = pd.DataFrame(lista_resultados)
    df_resultados.set_index('ID', inplace=True)
    df_resultados.to_csv('resultados.csv', encoding='utf-8') 