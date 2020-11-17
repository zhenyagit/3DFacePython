from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np
import skimage.transform as tr

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_template = np.load("face_template.npy")
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream, resize it to
	# have a maximum width of 400 pixels, and convert it to
	# grayscale
	man1 = "211.jpg"
	man2 = "22.jpg"
	frame1= cv2.imread(man1)
	frame2 = cv2.imread(man2)
	frame1 = cv2.resize(frame1, (750,1000), interpolation = cv2.INTER_AREA)
	frame2 = cv2.resize(frame2, (600,800), interpolation = cv2.INTER_AREA)
	gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
	gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
	
	# detect faces in the grayscale frame
	rects1 = detector(gray1, 0)
	rects2 = detector(gray2, 0)
	# check to see if a face was detected, and if so, draw the total

	def findshapes(rects,gray):
		for rect in rects:
			(bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)
		return shape
	test = np.array([[12,32],[13,312]])
	test = np.append(test,np.array([[44,45]]))
	shape1 = findshapes(rects1,gray1)
	shape2 = findshapes(rects2,gray2)
	def apdots(mas,k,num,dot):
		for i in range(k):
			return np.append(mas,np.array([[shape1[num+i][0]+int((dot[0]-shape1[num+i][0])/10), shape1[num+i][1]+int((dot[1]-shape1[num+i][1])/10)]]),axis=0)	
	for i in range(17):
		shape1 = np.append(shape1,np.array([[shape1[i][0]+int((shape1[30][0]-shape1[i][0])/10), shape1[i][1]+int((shape1[30][1]-shape1[i][1])/10)]]),axis=0)	
	for i in range(17):
		shape1 = np.append(shape1,np.array([[shape1[68+i][0]+int((shape1[30][0]-shape1[68+i][0])/10), shape1[68+i][1]+int((shape1[30][1]-shape1[68+i][1])/10)]]),axis=0)	
	for i in range(17):
		shape1 = np.append(shape1,np.array([[shape1[85+i][0]+int((shape1[30][0]-shape1[85+i][0])/10), shape1[85+i][1]+int((shape1[30][1]-shape1[85+i][1])/10)]]),axis=0)
	for i in range(5):
		shape1 = np.append(shape1,np.array([[shape1[102+i][0]+int((shape1[30][0]-shape1[102+i][0])/10), shape1[102+i][1]+int((shape1[30][1]-shape1[102+i][1])/10)]]),axis=0)
	for i in range(5):
		shape1 = np.append(shape1,np.array([[shape1[114+i][0]+int((shape1[30][0]-shape1[114+i][0])/10), shape1[114+i][1]+int((shape1[30][1]-shape1[114+i][1])/10)]]),axis=0)	
	for i in range(5):
		shape1 = np.append(shape1,np.array([[shape1[119+i][0]+int((shape1[30][0]-shape1[119+i][0])/9), shape1[119+i][1]+int((shape1[30][1]-shape1[119+i][1])/9)]]),axis=0)	
	for i in range(5):
		shape1 = np.append(shape1,np.array([[shape1[124+i][0]+int((shape1[30][0]-shape1[124+i][0])/9), shape1[124+i][1]+int((shape1[30][1]-shape1[124+i][1])/9)]]),axis=0)	
	for i in range(5):
		shape1 = np.append(shape1,np.array([[shape1[129+i][0]+int((shape1[30][0]-shape1[129+i][0])/6), shape1[129+i][1]+int((shape1[30][1]-shape1[129+i][1])/6)]]),axis=0)	
	for i in range(5):
		shape1 = np.append(shape1,np.array([[shape1[134+i][0]+int((shape1[30][0]-shape1[134+i][0])/6), shape1[134+i][1]+int((shape1[30][1]-shape1[134+i][1])/6)]]),axis=0)	
	for i in range(5):
		shape1 = np.append(shape1,np.array([[shape1[139+i][0]+int((shape1[30][0]-shape1[139+i][0])/6), shape1[139+i][1]+int((shape1[30][1]-shape1[139+i][1])/6)]]),axis=0)	
	for i in range(5):
		shape1 = np.append(shape1,np.array([[shape1[144+i][0]+int((shape1[30][0]-shape1[144+i][0])/6), shape1[144+i][1]+int((shape1[30][1]-shape1[144+i][1])/6)]]),axis=0)	
	for i in range(2):
		shape1 = np.append(shape1,np.array([[shape1[149+i][0]+int((shape1[30][0]-shape1[139+i][0])/6), shape1[139+i][1]+int((shape1[30][1]-shape1[139+i][1])/6)]]),axis=0)	
	for i in range(2):
		shape1 = np.append(shape1,np.array([[shape1[157+i][0]+int((shape1[30][0]-shape1[157+i][0])/6), shape1[157+i][1]+int((shape1[30][1]-shape1[157+i][1])/6)]]),axis=0)	
	shape1 = np.append(shape1,np.array([[shape1[153][0]+int((shape1[154][0]-shape1[153][0])/4), shape1[153][1]+int((shape1[154][1]-shape1[153][1])/4)]]),axis=0)
	shape1 = np.append(shape1,np.array([[shape1[154][0]+int((shape1[153][0]-shape1[154][0])/4), shape1[154][1]+int((shape1[153][1]-shape1[154][1])/4)]]),axis=0)
	print(len(shape1))
	shape1 = np.append(shape1,np.array([[shape1[163][0]+int((shape1[164][0]-shape1[163][0])/3), shape1[163][1]+int((shape1[164][1]-shape1[163][1])/3)]]),axis=0)
	shape1 = np.append(shape1,np.array([[shape1[164][0]+int((shape1[163][0]-shape1[164][0])/3), shape1[164][1]+int((shape1[163][1]-shape1[164][1])/3)]]),axis=0)
	for i in range(10):
		cv2.putText(frame1,str(len(shape1)-i-1), (shape1[len(shape1)-i-1][0], shape1[len(shape1)-i-1][1]),cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 255), 1)
	lche = (shape1[37]+shape1[38]+shape1[40]+shape1[41])/4
	cv2.circle(frame1, (int(lche[0]), int(lche[1])), 3, (150, 0, 255), -1)

	for i in shape1:
		cv2.circle(frame1, (int(i[0]), int(i[1])), 3, (150, 0, 255), -1)
	for i in shape2:
		cv2.circle(frame2, (int(i[0]), int(i[1])), 1, (150, 0, 255), -1)


	# INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]
	# proper_landmarks = 227 * face_template[INNER_EYES_AND_BOTTOM_LIP]+100
	# current_landmarks = landmarks[INNER_EYES_AND_BOTTOM_LIP]

	# A = np.hstack([current_landmarks, np.ones((3, 1))]).astype(np.float64)
	# print(A)
	# B = np.hstack([proper_landmarks, np.ones((3, 1))]).astype(np.float64)
	# print(B)
	# T = np.linalg.solve(A, B).T
	# print(T)
	# dots = np.dot(np.hstack([landmarks,np.ones((68,1))]).astype(np.float64),T)
	# wrapped = tr.warp(frame,tr.AffineTransform(T).inverse,output_shape=(400, 400),order=3,mode='constant',cval=0,clip=True,	preserve_range=True)
	# wrapped /= 255.0
	
	
	cv2.imshow("Frame", frame1)
	cv2.imshow("Frame2", frame2)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

cv2.destroyAllWindows()