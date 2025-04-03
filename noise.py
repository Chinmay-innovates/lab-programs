import cv2
import matplotlib.pyplot as plt

img = cv2.imread("C:/Users/CHINMAY/Downloads/noise.jpg",0)

if img is None:
    raise ValueError("Image not found. Check the file path.")

dst = cv2.fastNlMeansDenoising(img,None,10,7,21)

plt.subplot(121),plt.imshow(img,cmap='gray')
plt.subplot(122),plt.imshow(dst,cmap='gray')
plt.show()

