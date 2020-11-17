import cv2
import math
import time
class TDCam():
	"""docstring for 3DCam"""
	def __init__(self, source,l,r):
		OPENCV_VID_INPUT = {
			"0": 0,
			"cam1": "http://10.42.0.55:8080/video",
			"cam2": "http://10.42.0.219:8080/video",
			"non": None,
		}
		if OPENCV_VID_INPUT[source] != None:
			self.source = OPENCV_VID_INPUT[source]
			self.camcapture = cv2.VideoCapture(self.source)
		self.H = 0
		self.W = 0
		self.frame = 0	
		self.radius = r
		self.should = l
		self.xmouse = 10
		self.ymouse = 10
		# print(self.source)
	def mouse_moving(self,event, x, y, flags, params):
		if event == cv2.EVENT_MOUSEMOVE:
			self.xmouse = x
			self.ymouse = y
	def takeframe(self):
		self.camcapture.grab()
		self.camcapture.grab()
		self.camcapture.grab()
		ret, self.frame = self.camcapture.read()
		(self.H,self.W) = self.frame.shape[:2]
	def drawkrestik(self,razmer = 10,color = (0,0,255)):
		cv2.line(self.frame,(self.W/2-razmer,self.H/2),(self.W/2+razmer,self.H/2),color)
		cv2.line(self.frame,(self.W/2,self.H/2-razmer),(self.W/2,self.H/2+razmer),color)
	def angleged(self,k):
		k = (k-self.W/2)*self.should/self.W
		return (math.degrees(math.asin(k/math.sqrt(self.radius*self.radius-math.pow((self.should/2),2)+k*k))))
	def anglerad(self,k):
		k = (k-self.W/2)*self.should/self.W
		return (math.asin(k/math.sqrt(self.radius*self.radius-math.pow((self.should/2),2)+k*k)))
	def angleradvertical(self,k):
		k = (k-self.H/2)*self.should/self.H
		return (math.asin(k/math.sqrt(self.radius*self.radius-math.pow((self.should/2),2)+k*k)))
	# def angleradhead(self,(k1,k2),(n1,n2),lenth,s):
	# 	s1 = abs(math.sin(self.anglerad(k1))-math.sin(self.anglerad(n1)))
	# 	h1 = abs(math.sin(self.angleradvertical(k2))-math.sin(self.angleradvertical(n2)))
	# 	l1 = math.sqrt(s1*s1+h1*h1)*lenth
	# 	print(l1)
	# 	return math.acos(l1/s)
	@staticmethod
	def lengthh(k1,k2,s):
		return (s*math.cos(k1)*math.cos(-k2)/math.sin(k1-k2))
	def hight(self,k,l):
		angel = self.angleradvertical(k)
		return (math.tan(angel)*l)
	def weight(self,k,l):
		angel = self.anglerad(k)
		return (math.tan(angel)*l)
	def cropimg(self,img,x,y,rad=50):
		if x>=rad and y>=rad:
			return (img[y-rad:y+rad, x-rad:x+rad])
		else:
			return (img[self.H/2-rad:self.H/2+rad, self.W/2-rad:self.W/2+rad])
	def onlinecrop(self):
		return self.cropimg(self.frame,self.xmouse,self.ymouse)
	def debugg(self,namewindow):
		self.takeframe()
		self.drawkrestik()
		cv2.namedWindow(namewindow)
		cv2.setMouseCallback(namewindow, self.mouse_moving)
		info = [
				("x = ", self.xmouse),
				("y = ", self.ymouse),
				("angle = ", self.angleged(self.xmouse)),
			]
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(self.frame, text, (10, self.H - ((i * 20) + 20)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
		cv2.imshow(namewindow, self.frame)
	