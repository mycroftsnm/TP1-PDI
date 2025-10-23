import cv2
import numpy as np
import matplotlib.pyplot as plt

def ecualizacion_local(imagen, M=8, N=8):
   
    clahe = cv2.createCLAHE(clipLimit=15.0, tileGridSize=(M, N))
    imagen_eq = clahe.apply(imagen)

    return imagen_eq


img = cv2.imread("Imagen_con_detalles_escondidos.tif")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ventanas = [(10, 10),(12, 12), (16, 16), (21, 21)]

resultados = [ecualizacion_local(img_gray, M, N) for M, N in ventanas]

resultados_norm = [cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX) for res in resultados]





fig1, ax1 = plt.subplots()
ax1.imshow(img_gray, cmap='gray')
ax1.set_title('Original')
ax1.axis('off')
plt.show()


fig, axes = plt.subplots(2, 2, sharex=ax1, sharey=ax1, figsize=(12, 12))
axes = axes.flatten()

for i, (res, (M, N)) in enumerate(zip(resultados_norm, ventanas)):
    axes[i].imshow(res, cmap='gray')
    axes[i].set_title(f'Ventana {M}x{N}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()
