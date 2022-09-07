import cv2
import imutils

img1 = cv2.imread('photo.jpg',0)          # queryImage
img2 = cv2.imread('photo2.jpg',0) # trainImage

# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches ,None, flags=2)
img3 = imutils.resize(img3, width=1000)
while(1):
	cv2.imshow("asda", img3)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break