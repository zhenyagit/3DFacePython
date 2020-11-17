import numpy as np
import cv2
from matplotlib import pyplot as plt
# ceXuMH-NoSM.jpg
# xibDg06kgRY.jpg
imgR = cv2.imread('ceXuMH-NoSM.jpg',0)
imgL = cv2.imread('xibDg06kgRY.jpg',0)
# imgL = cv2.imread('21.jpg',0)
# imgR = cv2.imread('22.jpg',0)
imgR = cv2.resize(imgR,(600,800))
imgL = cv2.resize(imgL,(600,800))
stereo = cv2.StereoBM_create(numDisparities=208, blockSize=5)
disparity = stereo.compute(imgL,imgR)
cv2.namedWindow( "settings" ) # создаем окно настроек

if __name__ == '__main__':
	def nothing(*arg):
		pass

cv2.createTrackbar('h1', 'settings', 0, 255, nothing)
cv2.createTrackbar('s1', 'settings', 0, 20, nothing)
# while(1):
# 	h1 = cv2.getTrackbarPos('h1', 'settings')
# 	s1 = cv2.getTrackbarPos('s1', 'settings')
# 	try:
# 		stereo = cv2.StereoBM_create(numDisparities=h1, blockSize=s1)
# 		disparity = stereo.compute(imgL,imgR)
# 		print(h1)

# 	except Exception:
# 		pass
# 	cv2.imshow("wwww",disparity) 
# 	ch = cv2.waitKey(5)
# 	if ch == 27:
# 		break
plt.imshow(disparity,'gray')
plt.show()