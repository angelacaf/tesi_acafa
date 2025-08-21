# Algoritmo 1: Estrazione dei bordi dentali da immagini panoramiche

import cv2
import numpy as np

# Caricamento delle immagini panoramiche in scala di grigi
# px_reale.png  → immagine reale
# px_fake.png   → immagine sintetica generata da CycleGAN
img_real = cv2.imread('px_reale.png', cv2.IMREAD_GRAYSCALE)
img_fake = cv2.imread('px_fake.png', cv2.IMREAD_GRAYSCALE)

# Eventuale ridimensionamento per garantire stessa dimensione spaziale
if img_real.shape != img_fake.shape:
    img_fake = cv2.resize(img_fake, (img_real.shape[1], img_real.shape[0]))

# Applicazione del filtro di Canny per l'estrazione dei contorni dentali
edges_real = cv2.Canny(img_real, 50, 150)
edges_fake = cv2.Canny(img_fake, 50, 150)

# Visualizzazione dei contorni estratti (opzionale)
cv2.imshow("Contorni Reale", edges_real)
cv2.imshow("Contorni Sintetica", edges_fake)
cv2.waitKey(0)
cv2.destroyAllWindows()
