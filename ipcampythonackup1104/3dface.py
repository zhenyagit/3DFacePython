import stereocam
import cv2
import math
import imutils
import numpy as np
import dlib
from imutils import face_utils
xcam1 = 0;
ycam1 = 0
xcam2 = 0;
ycam2 = 0

# cam1 = stereocam.TDCam("0",1,3)
cam2 = stereocam.TDCam("0",46.5,40)
blank_image = np.zeros((480,640,3), np.uint8)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/zhenya/workdir/3dface/ipcampythonackup1104/shape_predictor_68_face_landmarks.dat')
i=100
j=200
while(1):
	blank_image = np.zeros((480,640,3), np.uint8)
	cam2.takeframe()
	gray1 = cv2.cvtColor(cam2.frame, cv2.COLOR_BGR2GRAY)
	rects1 = detector(gray1, 0)
	triangles = []
	for rect in rects1:
		
		shape = predictor(gray1, rect)
		shape = face_utils.shape_to_np(shape)
		koef =4.8
		for i in range(4):
			cv2.circle(cam2.frame, (shape[i+27][0], shape[i+27][1]), 1, (0, 0, 255), -1)
		cv2.circle(cam2.frame, (shape[8][0], shape[8][1]), 1, (0, 0, 255), -1)
		# cv2.line(cam2.frame, (shape[8][0], shape[8][1]), (shape[27][0], shape[27][1]),(0,255,0))
		# cv2.line(cam2.frame, (shape[27][0], shape[27][1]), (shape[30][0], shape[30][1]),(0,255,0))
		# cv2.line(cam2.frame, (int((shape[8][0]+(koef-1)*shape[27][0])/koef), int((shape[8][1]+(koef-1)*shape[27][1])/koef)), (shape[30][0], shape[30][1]),(0,255,0))
		
		lefteye = (int((shape[36][0]+shape[39][0])/2), int((shape[37][1]+shape[38][1]+shape[40][1]+shape[41][1])/4))
		rigteye = (int((shape[42][0]+shape[45][0])/2), int((shape[43][1]+shape[44][1]+shape[46][1]+shape[47][1])/4))
		nos = (shape[30][0], shape[30][1])
		cv2.line(cam2.frame, lefteye, rigteye,(0,255,0))
		cv2.line(cam2.frame, lefteye, nos,(0,255,0))
		cv2.line(cam2.frame, rigteye, nos,(0,255,0))
		cv2.line(blank_image, lefteye, rigteye,(0,255,0))
		cv2.line(blank_image, (0+i,0+j), (37+i,45+j),(0,255,0))
		cv2.line(blank_image, (74+i,0+j), (0+i,0+j),(0,255,0))
		cv2.line(blank_image, (37+i,45+j), (74+i,0+j),(0,255,0))
		cv2.line(blank_image, lefteye, nos,(0,255,0))
		cv2.line(blank_image, rigteye, nos,(0,255,0))
		cv2.putText(cam2.frame, str(lefteye[0])+" "+str(lefteye[1]), lefteye, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
		cv2.putText(cam2.frame, str(rigteye[0])+" "+str(rigteye[1]), rigteye, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
		cv2.putText(cam2.frame, str(nos[0])+" "+str(nos[1]), nos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
		
		cv2.circle(cam2.frame, lefteye, 1, (0, 0, 255), -1)
		cv2.circle(cam2.frame, rigteye, 1, (0, 0, 255), -1)
		cv2.circle(cam2.frame, nos, 1, (0, 0, 255), -1)
		triangles.append([lefteye,rigteye,nos])
		
		xcam1 = shape[30][0]
	

	cv2.imshow("Frame1", cam2.frame)
	cv2.imshow("Frame2", blank_image)
	
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
