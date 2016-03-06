import cv2
import numpy as np


WHITE = (255,255,255)


class ColourDetector:

	# detected colours list
	colours = []

	# colours table
	_colour_table = [
		("white",	(255,255,255)),
		("black",	(0,0,0)),
		("red",		(0,0,255)),
		("green",	(0,255,0)),
		("blue",	(255,0,0)),
		("yellow",	(0,255,255)),
		("orange",	(0,128,255)),
		#("cyan",	(255,255,0)),
		#("magenta",(255,0,255)),
	]

	# convert colours table from BGR to CIELAB colour space
	_colour_lab = np.array([c[1] for c in _colour_table]) \
					.reshape(len(_colour_table),1,3) \
					.astype(np.uint8)
	_colour_lab = cv2.cvtColor(_colour_lab, cv2.COLOR_BGR2LAB)

	def __init__(self, contours, img):
		# convert image to CIELAB colour space
		cielab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

		# create list of colour labels
		self.colours = [self._detect_colour(cnt, cielab_img)
			for cnt in contours]

	def _detect_colour(self, cnt, img):
		# create a mask for each contour
		mask = np.zeros(img.shape[:2], dtype=np.uint8)
		cv2.drawContours(mask, [cnt], 0, WHITE, -1)
		mask = cv2.erode(mask, None, iterations=2)

		# calculate mean CIELAB colour of each contour
		mean_colour = cv2.mean(img, mask=mask)[:3]

		# search for colour in colour table
		idx = np.argmin([np.linalg.norm(mean_colour - c)
			for c in self._colour_lab])
		return self._colour_table[idx][0]
