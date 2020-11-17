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
cam1 = stereocam.TDCam("non",900,1000)
cam2 = stereocam.TDCam("non",900,1000)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
man1 = "21.jpg"
man2 = "22.jpg"
cam1.frame = cv2.imread(man1)
cam1.frame = cv2.resize(cam1.frame, (600,800), interpolation = cv2.INTER_AREA)
cam2.frame = cv2.imread(man2)
cam2.frame = cv2.resize(cam2.frame, (600,800), interpolation = cv2.INTER_AREA)
(cam1.H,cam1.W) = cam1.frame.shape[:2]
(cam2.H,cam2.W) = cam2.frame.shape[:2]
gray1 = cv2.cvtColor(cam1.frame, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(cam2.frame, cv2.COLOR_BGR2GRAY)
rects1 = detector(gray1, 0)
rects2 = detector(gray2, 0)

# newim = np.zeros((hight,widht,3), np.uint8)

def dotintwo(rects, image):
	# widht = 800
	# hight = 800
	# newim = np.zeros((hight,widht,3), np.uint8)
	for rect in rects:
		shape = predictor(image, rect)
		shape = face_utils.shape_to_np(shape)
	return shape

def alldottree(cam1,l,dot1,dot2):
	dotin3 = []
	#########################################################
	for i in range(len(dot1)):
		z = stereocam.TDCam.lengthh(cam1.anglerad(dot1[i][0]),cam2.anglerad(dot2[i][0]),l)
		x = int(cam1.weight(dot1[i][0],z))
		y = int(cam1.weight(dot1[i][1],z))
		z = int(z)
		dotin3.append([x,y,z])
	return dotin3

dots1 = dotintwo(rects1,cam1.frame)
dots2 = dotintwo(rects2,cam2.frame)

for i in range(17):
	dots1 = np.append(dots1,np.array([[dots1[i][0]+int((dots1[30][0]-dots1[i][0])/10), dots1[i][1]+int((dots1[30][1]-dots1[i][1])/10)]]),axis=0)	
for i in range(17):
	dots1 = np.append(dots1,np.array([[dots1[68+i][0]+int((dots1[30][0]-dots1[68+i][0])/10), dots1[68+i][1]+int((dots1[30][1]-dots1[68+i][1])/10)]]),axis=0)		
for i in range(17):
	dots1 = np.append(dots1,np.array([[dots1[85+i][0]+int((dots1[30][0]-dots1[85+i][0])/10), dots1[85+i][1]+int((dots1[30][1]-dots1[85+i][1])/10)]]),axis=0)
for i in range(5):
	dots1 = np.append(dots1,np.array([[dots1[102+i][0]+int((dots1[30][0]-dots1[102+i][0])/10), dots1[102+i][1]+int((dots1[30][1]-dots1[102+i][1])/10)]]),axis=0)
for i in range(5):
	dots1 = np.append(dots1,np.array([[dots1[114+i][0]+int((dots1[30][0]-dots1[114+i][0])/10), dots1[114+i][1]+int((dots1[30][1]-dots1[114+i][1])/10)]]),axis=0)	
for i in range(5):
	dots1 = np.append(dots1,np.array([[dots1[119+i][0]+int((dots1[30][0]-dots1[119+i][0])/9), dots1[119+i][1]+int((dots1[30][1]-dots1[119+i][1])/9)]]),axis=0)	
for i in range(5):
	dots1 = np.append(dots1,np.array([[dots1[124+i][0]+int((dots1[30][0]-dots1[124+i][0])/9), dots1[124+i][1]+int((dots1[30][1]-dots1[124+i][1])/9)]]),axis=0)	
for i in range(5):
	dots1 = np.append(dots1,np.array([[dots1[129+i][0]+int((dots1[30][0]-dots1[129+i][0])/6), dots1[129+i][1]+int((dots1[30][1]-dots1[129+i][1])/6)]]),axis=0)	
for i in range(5):
	dots1 = np.append(dots1,np.array([[dots1[134+i][0]+int((dots1[30][0]-dots1[134+i][0])/6), dots1[134+i][1]+int((dots1[30][1]-dots1[134+i][1])/6)]]),axis=0)	
for i in range(5):
	dots1 = np.append(dots1,np.array([[dots1[139+i][0]+int((dots1[30][0]-dots1[139+i][0])/6), dots1[139+i][1]+int((dots1[30][1]-dots1[139+i][1])/6)]]),axis=0)	
for i in range(5):
	dots1 = np.append(dots1,np.array([[dots1[144+i][0]+int((dots1[30][0]-dots1[144+i][0])/6), dots1[144+i][1]+int((dots1[30][1]-dots1[144+i][1])/6)]]),axis=0)	
for i in range(2):
	dots1 = np.append(dots1,np.array([[dots1[149+i][0]+int((dots1[30][0]-dots1[139+i][0])/6), dots1[139+i][1]+int((dots1[30][1]-dots1[139+i][1])/6)]]),axis=0)	
for i in range(2):
	dots1 = np.append(dots1,np.array([[dots1[157+i][0]+int((dots1[30][0]-dots1[157+i][0])/6), dots1[157+i][1]+int((dots1[30][1]-dots1[157+i][1])/6)]]),axis=0)	
dots1 = np.append(dots1,np.array([[dots1[153][0]+int((dots1[154][0]-dots1[153][0])/4), dots1[153][1]+int((dots1[154][1]-dots1[153][1])/4)]]),axis=0)
dots1 = np.append(dots1,np.array([[dots1[154][0]+int((dots1[153][0]-dots1[154][0])/4), dots1[154][1]+int((dots1[153][1]-dots1[154][1])/4)]]),axis=0)
# print(len(dots1))
dots1 = np.append(dots1,np.array([[dots1[163][0]+int((dots1[164][0]-dots1[163][0])/3), dots1[163][1]+int((dots1[164][1]-dots1[163][1])/3)]]),axis=0)
dots1 = np.append(dots1,np.array([[dots1[164][0]+int((dots1[163][0]-dots1[164][0])/3), dots1[164][1]+int((dots1[163][1]-dots1[164][1])/3)]]),axis=0)

for i in range(17):
	dots2 = np.append(dots2,np.array([[dots2[i][0]+int((dots2[30][0]-dots2[i][0])/10), dots2[i][1]+int((dots2[30][1]-dots2[i][1])/10)]]),axis=0)	
for i in range(17):
	dots2 = np.append(dots2,np.array([[dots2[68+i][0]+int((dots2[30][0]-dots2[68+i][0])/10), dots2[68+i][1]+int((dots2[30][1]-dots2[68+i][1])/10)]]),axis=0)		
for i in range(17):
	dots2 = np.append(dots2,np.array([[dots2[85+i][0]+int((dots2[30][0]-dots2[85+i][0])/10), dots2[85+i][1]+int((dots2[30][1]-dots2[85+i][1])/10)]]),axis=0)
for i in range(5):
	dots2 = np.append(dots2,np.array([[dots2[102+i][0]+int((dots2[30][0]-dots2[102+i][0])/10), dots2[102+i][1]+int((dots2[30][1]-dots2[102+i][1])/10)]]),axis=0)
for i in range(5):
	dots2 = np.append(dots2,np.array([[dots2[114+i][0]+int((dots2[30][0]-dots2[114+i][0])/10), dots2[114+i][1]+int((dots2[30][1]-dots2[114+i][1])/10)]]),axis=0)	
for i in range(5):
	dots2 = np.append(dots2,np.array([[dots2[119+i][0]+int((dots2[30][0]-dots2[119+i][0])/9), dots2[119+i][1]+int((dots2[30][1]-dots2[119+i][1])/9)]]),axis=0)	
for i in range(5):
	dots2 = np.append(dots2,np.array([[dots2[124+i][0]+int((dots2[30][0]-dots2[124+i][0])/9), dots2[124+i][1]+int((dots2[30][1]-dots2[124+i][1])/9)]]),axis=0)	
for i in range(5):
	dots2 = np.append(dots2,np.array([[dots2[129+i][0]+int((dots2[30][0]-dots2[129+i][0])/6), dots2[129+i][1]+int((dots2[30][1]-dots2[129+i][1])/6)]]),axis=0)	
for i in range(5):
	dots2 = np.append(dots2,np.array([[dots2[134+i][0]+int((dots2[30][0]-dots2[134+i][0])/6), dots2[134+i][1]+int((dots2[30][1]-dots2[134+i][1])/6)]]),axis=0)	
for i in range(5):
	dots2 = np.append(dots2,np.array([[dots2[139+i][0]+int((dots2[30][0]-dots2[139+i][0])/6), dots2[139+i][1]+int((dots2[30][1]-dots2[139+i][1])/6)]]),axis=0)	
for i in range(5):
	dots2 = np.append(dots2,np.array([[dots2[144+i][0]+int((dots2[30][0]-dots2[144+i][0])/6), dots2[144+i][1]+int((dots2[30][1]-dots2[144+i][1])/6)]]),axis=0)	
for i in range(2):
	dots2 = np.append(dots2,np.array([[dots2[149+i][0]+int((dots2[30][0]-dots2[139+i][0])/6), dots2[139+i][1]+int((dots2[30][1]-dots2[139+i][1])/6)]]),axis=0)	
for i in range(2):
	dots2 = np.append(dots2,np.array([[dots2[157+i][0]+int((dots2[30][0]-dots2[157+i][0])/6), dots2[157+i][1]+int((dots2[30][1]-dots2[157+i][1])/6)]]),axis=0)	
dots2 = np.append(dots2,np.array([[dots2[153][0]+int((dots2[154][0]-dots2[153][0])/4), dots2[153][1]+int((dots2[154][1]-dots2[153][1])/4)]]),axis=0)
dots2 = np.append(dots2,np.array([[dots2[154][0]+int((dots2[153][0]-dots2[154][0])/4), dots2[154][1]+int((dots2[153][1]-dots2[154][1])/4)]]),axis=0)
# print(len(dots2))
dots2 = np.append(dots2,np.array([[dots2[163][0]+int((dots2[164][0]-dots2[163][0])/3), dots2[163][1]+int((dots2[164][1]-dots2[163][1])/3)]]),axis=0)
dots2 = np.append(dots2,np.array([[dots2[164][0]+int((dots2[163][0]-dots2[164][0])/3), dots2[164][1]+int((dots2[163][1]-dots2[164][1])/3)]]),axis=0)
mass = alldottree(cam1,77,dots1,dots2)
def normalize(mass):
	minn = mass[0][2]
	maxx = mass[86][2]
	for dot in mass:
		# if dot[2]>maxx:
		# 	maxx = dot[2]
		if dot[2]<minn:
			minn = dot[2]
	delta = maxx-minn
	newmass = []
	for dot in mass:
		newmass.append((dot[2]-minn)/delta)
	return newmass
normalizetz = normalize(mass)
# print(normalizetz)
for i in range(len(dots1)):
	cv2.circle(cam1.frame, (dots1[i][0], dots1[i][1]), 3, (0, int(255-255*normalizetz[i]),int(255*normalizetz[i]) ), -1)
	# cv2.putText(cam1.frame,str(int((mass[i][2]-300)/10)), (dots1[i][0], dots1[i][1]),cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 1)
while(1):
	cv2.imshow("Frame1", cam1.frame)
	cv2.imshow("Frame2", cam2.frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
