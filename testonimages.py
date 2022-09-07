from stereocam import Cam,StereoCam
import cv2
import numpy as np
import dlib
from imutils import face_utils
from line_profiler_pycharm import profile
@profile
def main():
	images_size = (600,800)
	cams =[]
	cams.append(Cam(900, 1000, None))
	cams.append(Cam(900, 1000, None))
	images_path = ["21.jpg", "22.jpg"]
	images_dots = np.zeros(shape=(2,68,2))
	images_rects = []
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

	for index, (cam, img_path) in enumerate(zip(cams,images_path)):
		cam.set_frame(cv2.resize(cv2.imread(img_path), images_size, interpolation=cv2.INTER_AREA))
		images_dots[index] = get_face_dots_one(detector(cv2.cvtColor(cam.frame, cv2.COLOR_BGR2GRAY), 0), cam.frame, predictor)

	stereo_cam = StereoCam(*cams, width_between=77)
	distances = stereo_cam.calc_distances(images_dots)
	normalized = normalize(distances)
	show_dots_distance(cams, images_dots, normalized)

def get_face_dots_one(rects, image, predictor):
	return face_utils.shape_to_np(predictor(image, rects[0]))

def get_face_dots(rects, image, predictor):
	shape = None
	for rect in rects:
		shape = predictor(image, rect)
		shape = face_utils.shape_to_np(shape)
	return shape

def normalize(mass):
	mass = mass - np.min(mass)
	mass = mass/np.max(mass)
	return mass
@profile
def show_dots_distance(cams, dots_lr, distances):
	for index, (cam, dots) in enumerate(zip(cams, dots_lr)):
		for index, (dot, distance) in enumerate(zip(dots, distances)):
			cv2.circle(cam.frame, (int(dot[0]), int(dot[1])), 3, (0, int(255-255*distance), int(255*distance)), -1)
	while 1:
		cv2.imshow("Frame left", cams[0].frame)
		cv2.imshow("Frame right", cams[1].frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break


if __name__ =="__main__":
	main()