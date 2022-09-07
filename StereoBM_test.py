import cv2
from matplotlib import pyplot as plt
imgR = cv2.imread('ceXuMH-NoSM.jpg',0)
imgL = cv2.imread('xibDg06kgRY.jpg',0)
imgR = cv2.resize(imgR,(600,800))
imgL = cv2.resize(imgL,(600,800))
stereo = cv2.StereoBM_create(numDisparities=16*11, blockSize=5)
disparity = stereo.compute(imgL,imgR)

plt.imshow(disparity,'gray')
plt.show()