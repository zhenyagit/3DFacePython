from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np
import skimage.transform as tr

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
args = vars(ap.parse_args())

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
 
# initialize the video stream and sleep for a bit, allowing the
# camera sensor to warm up
print("[INFO] camera sensor warming up...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start() # Raspberry Pi
time.sleep(2.0)
face_template = np.load("face_template.npy")
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream, resize it to
	# have a maximum width of 400 pixels, and convert it to
	# grayscale
	man = "21.jpg"
	man2 = "22.jpg"
	frame = cv2.imread(man)
	frame2 = cv2.imread(man2)
	frame = cv2.resize(frame, (600,800), interpolation = cv2.INTER_AREA)
	frame2 = cv2.resize(frame2, (600,800), interpolation = cv2.INTER_AREA)
	gray = cv2.imread(man,0)
	gray2 = cv2.imread(man2,0)
	gray = cv2.resize(gray, (600,800), interpolation = cv2.INTER_AREA)
	gray2 = cv2.resize(gray2, (600,800), interpolation = cv2.INTER_AREA)
	# detect faces in the grayscale frame
	rects = detector(gray, 0)
	rects2 = detector(gray2, 0)
	# check to see if a face was detected, and if so, draw the total
	# number of faces on the frame
	if len(rects) > 0:
		text = "{} face(s) found".format(len(rects))
		cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, (0, 0, 255), 2)
	if len(rects2) > 0:
		text = "{} face(s) found".format(len(rects2))
		cv2.putText(frame2, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
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
		landmarks = np.array(list(map(lambda p: [p.x, p.y], shape.parts())))
		shape = face_utils.shape_to_np(shape)
		
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw each of them
		for (i, (x, y)) in enumerate(shape):
			cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
			# cv2.putText(frame, str(i + 1), (x - 10, y - 10),
			# 	cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
				# show the frame
	for rect in rects2:
		# compute the bounding box of the face and draw it on the
		# frame
		(bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
		# cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH),
		# 	(0, 255, 0), 1)
 
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray2, rect)
		shape = face_utils.shape_to_np(shape)
 
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw each of them
		for (i, (x, y)) in enumerate(shape):
			cv2.circle(frame2, (x, y), 1, (0, 0, 255), -1)
			# cv2.putText(frame, str(i + 1), (x - 10, y - 10),
			# 	cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
				# show the frame
	INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]
	proper_landmarks = 227 * face_template[INNER_EYES_AND_BOTTOM_LIP]+100
	current_landmarks = landmarks[INNER_EYES_AND_BOTTOM_LIP]

	A = np.hstack([current_landmarks, np.ones((3, 1))]).astype(np.float64)
	print(A)
	B = np.hstack([proper_landmarks, np.ones((3, 1))]).astype(np.float64)
	print(B)
	T = np.linalg.solve(A, B).T
	print(T)
	dots = np.dot(np.hstack([landmarks,np.ones((68,1))]).astype(np.float64),T)
	wrapped = tr.warp(frame,tr.AffineTransform(T).inverse,output_shape=(400, 400),order=3,mode='constant',cval=0,clip=True,	preserve_range=True)
	wrapped /= 255.0
	
	for i in dots:
		cv2.circle(wrapped, (int(i[0]), int(i[1])), 1, (0, 255, 255), -1)
	cv2.imshow("FrameWraped", wrapped)
	cv2.imshow("Frame", frame)
	cv2.imshow("Frame2", frame2)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()