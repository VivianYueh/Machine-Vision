import cv2
import matplotlib.pyplot as plt
import numpy as np

frame1=cv2.imread("a.jpg")
B=frame1[:,:,0]
G=frame1[:,:,1]
R=frame1[:,:,0]
equalized1 = cv2.equalizeHist(frame1)
equalized_bgr1 = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)
equalized_bgr1 = cv2.putText(equalized_bgr1, 'apply histogram equalization', (10,50), cv2.FONT_HERSHEY_SIMPLEX,  2, (255,255,255), 2, cv2.LINE_AA) 
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(equalized_bgr1.astype(np.uint8))
plt.title('Grayscale')
plt.axis('off')
plt.show()