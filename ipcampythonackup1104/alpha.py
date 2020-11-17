import stereocam
import cv2
import math
import imutils
import numpy as np
import dlib
from imutils import face_utils

# cam1 = stereocam.TDCam("0",1,3)
cam2 = stereocam.TDCam("cam1",46.5,40)
# cam3 = stereocam.TDCam("cam2",40.5,40)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/zhenya/workdir/3dface/ipcampythonackup1104/shape_predictor_68_face_landmarks.dat')

while(1):
	# cam1.debugg("frame")

	cam2.debugg("frame2")
	# miniframe = cam2.onlinecrop()
	gray = cv2.cvtColor(cam2.frame, cv2.COLOR_BGR2GRAY)
 
	# detect faces in the grayscale frame
	rects = detector(gray, 0)
 
	# check to see if a face was detected, and if so, draw the total
	# number of faces on the frame
	if len(rects) > 0:
		text = "{} face(s) found".format(len(rects))
		cv2.putText(cam2.frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, (0, 0, 255), 2)
	# loop over the face detections
	for rect in rects:
		# compute the bounding box of the face and draw it on the
		# frame
		(bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
		# cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH),
		# 	(0, 255, 0), 1)
 
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
 
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw each of them
		koef = 5.1
		for i in range(4):
			cv2.circle(cam2.frame, (shape[i+27][0], shape[i+27][1]), 1, (0, 0, 255), -1)
		cv2.circle(cam2.frame, (shape[8][0], shape[8][1]), 1, (0, 0, 255), -1)
		cv2.line(cam2.frame, (shape[8][0], shape[8][1]), (shape[27][0], shape[27][1]),(0,255,0))
		cv2.line(cam2.frame, (shape[27][0], shape[27][1]), (shape[30][0], shape[30][1]),(0,255,0))
		cv2.line(cam2.frame, (int((shape[8][0]+(koef-1)*shape[27][0])/koef), int((shape[8][1]+(koef-1)*shape[27][1])/koef)), (shape[30][0], shape[30][1]),(0,255,0))
		
	# cv2.imshow("asda", miniframe)
	cv2.imshow("Frame222", cam2.frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
