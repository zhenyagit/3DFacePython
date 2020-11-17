import cv2
import numpy as np

class Prostran(object):
	"""docstring for Prostran"""
	def __init__(self):
		self.xx = 300
		self.yy = 400
		self.zz = 250
		self.reb = 3
		self.scamm = 15
		self.ycamm = 20
		self.hcamm = int(self.zz/2)
		self.xcamm = int(self.xx/2)
		self.korcam1 = np.array([self.xcamm-int(self.scamm/2),self.ycamm,self.hcamm])
		self.korcam2 = np.array([self.xcamm+int(self.scamm/2),self.ycamm,self.hcamm])
		self.xozim = np.zeros((self.zz+1, self.xx+1, 3), np.uint8)
		self.yozim = np.zeros((self.zz+1, self.yy+1, 3), np.uint8)
		self.yoxim = np.zeros((self.yy+1, self.xx+1, 3), np.uint8)
		self.white = np.zeros((self.yy+1, self.yy+1, 3), np.uint8)
		self.xozim[:] = (255,255,255)
		self.yozim[:] = (255,255,255)
		self.yoxim[:] = (255,255,255)
		cv2.rectangle(self.xozim,(0+self.reb,0+self.reb),(self.xx-self.reb,self.zz-self.reb),(0,250,130))
		cv2.rectangle(self.yozim,(0+self.reb,0+self.reb),(self.yy-self.reb,self.zz-self.reb),(0,250,130))
		cv2.rectangle(self.yoxim,(0+self.reb,0+self.reb),(self.xx-self.reb,self.yy-self.reb),(0,250,130))

	def obedinen(self):
		lp1 = np.concatenate((self.xozim, self.yoxim), axis=0)
		lp2 = np.concatenate((self.yozim, self.white), axis=0)
		full = np.concatenate((lp1, lp2), axis=1)
		self.xozim[:] = (255,255,255)
		self.yozim[:] = (255,255,255)
		self.yoxim[:] = (255,255,255)
		return full
	
	def proection(self,plosk,dot):
		if plosk == 1: return(dot[0],dot[2]) #"xoz"
		if plosk == 2: return(self.yy-dot[1],dot[2]) #"yoz"
		if plosk == 3: return(dot[0],self.yy-dot[1]) #"yox" 
	
	def dotimag(self,koor,rad=2,color=(25,255,25),tol=-1):
		cv2.circle(self.xozim,self.proection(1,koor),rad,color,tol)
		cv2.circle(self.yozim,self.proection(2,koor),rad,color,tol)
		cv2.circle(self.yoxim,self.proection(3,koor),rad,color,tol) 
	
# k  = Prostran()
# dot = [0,10,100]
# while(1):
# 	print(k.korcam1)
# 	print(k.korcam2)
	
# 	k.dotimag(dot,tol=-1)
# 	k.dotimag(k.korcam1,5,(24,24,255))
# 	k.dotimag(k.korcam2,5,(24,24,255))
# 	frame = k.obedinen()

# 	cv2.imshow("Frame4",frame)
	
# 	key = cv2.waitKey(1) & 0xFF
# 	if key == ord("q"):
# 		break
