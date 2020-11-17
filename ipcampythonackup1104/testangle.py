import stereocam
import imgdro
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
xreye2 = 0;
xreye1 = 0;
xleye1 = 0;
xleye2 = 0;
yreye2 = 0;
yreye1 = 0;
yleye1 = 0;
yleye2 = 0;
lefttrid = [0,0,0]
rigttrid = [0,0,0]

# cam1 = stereocam.TDCam("0",1,3)
postr = imgdro.Prostran()
cam2 = stereocam.TDCam("0",46.5,40)
# cam3 = stereocam.TDCam("cam2",40.5,40)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/zhenya/workspace/shape_predictor_68_face_landmarks.dat')

while(1):
	# cam1.debugg("frame")

	# cam2.debugg("frame2")
	
	cam2.takeframe()
	gray1 = cv2.cvtColor(cam2.frame, cv2.COLOR_BGR2GRAY)
	
	rects1 = detector(gray1, 0)
	for rect in rects1:
		
		shape = predictor(gray1, rect)
		shape = face_utils.shape_to_np(shape)
		koef =4.8
		# for i in range(4):
		# 	cv2.circle(cam2.frame, (shape[i+27][0], shape[i+27][1]), 1, (0, 0, 255), -1)
		# cv2.circle(cam2.frame, (shape[8][0], shape[8][1]), 1, (0, 0, 255), -1)
		# cv2.line(cam2.frame, (shape[8][0], shape[8][1]), (shape[27][0], shape[27][1]),(0,255,0))
		# cv2.line(cam2.frame, (shape[27][0], shape[27][1]), (shape[30][0], shape[30][1]),(0,255,0))
		# cv2.line(cam2.frame, (int((shape[8][0]+(koef-1)*shape[27][0])/koef), int((shape[8][1]+(koef-1)*shape[27][1])/koef)), (shape[30][0], shape[30][1]),(0,255,0))
		cv2.circle(cam2.frame, (shape[30][0], shape[30][1]), 1, (0, 0, 255), -1)
		lefteye1 = (int((shape[36][0]+shape[39][0])/2), int((shape[37][1]+shape[38][1]+shape[40][1]+shape[41][1])/4))
		rigteye1 = (int((shape[42][0]+shape[45][0])/2), int((shape[43][1]+shape[44][1]+shape[46][1]+shape[47][1])/4))
		cv2.circle(cam2.frame, lefteye1, 1, (0, 255, 0), -1)
		cv2.circle(cam2.frame, rigteye1, 1, (0, 255, 0), -1)
		xreye1 = int((shape[36][0]+shape[39][0])/2)
		xleye1 = int((shape[42][0]+shape[45][0])/2)
		yleye1 = int((shape[43][1]+shape[44][1]+shape[46][1]+shape[47][1])/4)
		yreye1 = int((shape[37][1]+shape[38][1]+shape[40][1]+shape[41][1])/4)
		xcam1 = shape[30][0]
		ycam1 = shape[30][1]
	

	lenas = 30
	
	angleheada =  cam2.angleradhead((xleye1,yleye1),(xreye1,yreye1),lenas,6.1)
	
	lefttrid[0] = int(cam2.weight(xleye1,lenas)) 
	lefttrid[2] = int(cam2.hight(yleye1,lenas))
	lefttrid[1] = int(lenas)
	rigttrid[0] = int(cam2.weight(xreye1,lenas))	
	rigttrid[2] = int(cam2.hight(yreye1,lenas))	
	rigttrid[1] = int(lenas)
	
	cv2.putText(cam2.frame, str(xcam1)+ " " +str(xcam2) + " length = " + str(lenas) , ( 10, 100),cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 0, 255), 1)
	cv2.putText(cam2.frame,"angel= " + str(math.degrees(angleheada)), ( 10, 120),cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 0, 255), 1)
	lefttrid[0] = postr.korcam1[0]+lefttrid[0]
	lefttrid[1] = postr.korcam1[1]+lefttrid[1]
	lefttrid[2] = postr.korcam1[2]+lefttrid[2]
	rigttrid[0] = postr.korcam1[0]+rigttrid[0]
	rigttrid[1] = postr.korcam1[1]+rigttrid[1]
	rigttrid[2] = postr.korcam1[2]+rigttrid[2]


	postr.dotimag(lefttrid,tol=-1)
	postr.dotimag(rigttrid,tol=-1)

	postr.dotimag(postr.korcam1,5,(24,24,255))
	postr.dotimag(postr.korcam2,5,(24,24,255))
	frame = postr.obedinen()

	cv2.imshow("Frame1", cam2.frame)
	cv2.imshow("Frame3", frame)

	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
