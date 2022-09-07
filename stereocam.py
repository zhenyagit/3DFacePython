import cv2
import time
import numpy as np
from prettytable import PrettyTable


class Cam:
	def __init__(self, width_of_frame, radius, source=0):
		print("Init Cam")
		self.frame = None
		self.W = 0
		self.H = 0
		self.radius = radius
		self.width_of_frame = width_of_frame
		self.x_mouse = 10
		self.y_mouse = 10
		self.viewing_angle = self.calc_viewing_angle(width_of_frame, radius)
		self.distance_to_chord = self.calc_distance_to_chord(width_of_frame, radius)

		if source is not None:
			self.source = source
			self.camcapture = cv2.VideoCapture(self.source)
			self.take_frame()
			(self.H, self.W) = self.frame.shape[:2]
			self.center_of_image = np.array([int(self.W / 2), int(self.H / 2)])
		self.show_info()
		print("Init done!")

	def show_info(self):
		x = PrettyTable()
		x.field_names = ["Parameter", "Value"]
		x.add_row(["w,h", str([self.W, self.H])])
		x.add_row(["radius", self.radius])
		x.add_row(["width_of_frame", self.width_of_frame])
		x.add_row(["viewing_angle", np.degrees(self.viewing_angle)])
		x.add_row(["distance_to_chord", self.distance_to_chord])
		print(x)

	@staticmethod
	def calc_viewing_angle(width_of_frame, radius):
		cathetus = width_of_frame / 2
		return np.arcsin(cathetus / radius) * 2

	@staticmethod
	def calc_distance_to_chord(width_of_frame, radius):
		return np.sqrt(radius ** 2 - (width_of_frame / 2) ** 2)

	def mouse_moving(self, event, x, y, flags, params):
		if event == cv2.EVENT_MOUSEMOVE:
			self.x_mouse = x
			self.y_mouse = y

	def take_frame(self):
		# self.camcapture.grab()
		_, self.frame = self.camcapture.read()

	def set_frame(self, frame):
		self.frame = frame
		(self.H, self.W) = self.frame.shape[:2]
		self.center_of_image = np.array([int(self.W / 2), int(self.H / 2), ])

	def draw_crosshair(self, radius=10, color=(0, 0, 255)):
		x = np.array([radius, 0])
		y = np.array([0, radius])
		cv2.line(self.frame, tuple(self.center_of_image - x), tuple(self.center_of_image + x), color)
		cv2.line(self.frame, tuple(self.center_of_image - y), tuple(self.center_of_image + y), color)

	def angle_from_center_rad(self, x_pixels):
		from_left_to_right = self.width_of_frame * x_pixels / self.W
		from_centre = from_left_to_right - self.width_of_frame / 2
		return np.arctan(from_centre / self.distance_to_chord)

	def angle_from_center_deg(self, x_pixels):
		return np.degrees(self.angle_from_center_rad(x_pixels))

	def angle_from_center_rad_v(self, y_pixels):
		if y_pixels < 0 or y_pixels > self.H:
			return None
		from_centre_pixels = y_pixels - self.H / 2
		from_centre_real = self.width_of_frame * from_centre_pixels / self.W
		return np.arctan(from_centre_real / self.distance_to_chord)

	def angle_from_center_deg_v(self, y_pixels):
		return np.degrees(self.angle_from_center_rad_v(y_pixels))

	@staticmethod
	def crop_img(self, img, x, y, rad=50):
		if x >= rad and y >= rad:
			return img[y - rad:y + rad, x - rad:x + rad]
		else:
			return img[self.H / 2 - rad:self.H / 2 + rad, self.W / 2 - rad:self.W / 2 + rad]

	def crop_img_where_mouse(self, rad=50):
		return self.crop_img(self.frame, self.x_mouse, self.y_mouse, rad)

	def demo(self, namewindow):
		cv2.namedWindow(namewindow)
		cv2.setMouseCallback(namewindow, self.mouse_moving)
		start = time.time()
		last10 = [60]
		while 1:
			self.take_frame()
			self.draw_crosshair()
			fps = 1 / (time.time() - start)
			last10.append(fps)
			if len(last10) > 10:
				last10.pop(0)
			start = time.time()
			info = [
				("fps = ", fps),
				("fps_avg = ", sum(last10) / len(last10)),
				("x = ", self.x_mouse),
				("y = ", self.y_mouse),
				("angle_x = ", self.angle_from_center_deg(self.x_mouse)),
				("angle_y = ", self.angle_from_center_deg_v(self.y_mouse)),
			]
			for (i, (k, v)) in enumerate(info):
				text = "{}: {}".format(k, v)
				cv2.putText(self.frame, text, (10, self.H - ((i * 20) + 20)),
							cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
			cv2.imshow(namewindow, self.frame)

			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break
		cv2.destroyAllWindows()


class StereoCam:
	def __init__(self, cam_l: Cam, cam_r: Cam, width_between):
		self.cam_l = cam_l
		self.cam_r = cam_r
		self.width_between = width_between

	def calc_distance_one_dot(self, dot_l, dot_r):
		angle_l = self.cam_l.angle_from_center_rad(dot_l[0])
		angle_r = self.cam_r.angle_from_center_rad(dot_r[0])
		return angle_l ^ 2 * (1 / np.sin(180 - angle_l - angle_r)) * np.sin(angle_l) * np.sin(angle_r) / self.width_between

	def calc_distances(self, dots):
		angle_l = -self.cam_l.angle_from_center_rad(dots[0][:,0])
		angle_r = self.cam_r.angle_from_center_rad(dots[1][:,0])
		return self.width_between*np.cos(angle_l)*np.cos(angle_r)/np.sin(angle_l+angle_r)


if __name__ == "__main__":
	a = Cam(200, 150)
	a.demo("Angle")
