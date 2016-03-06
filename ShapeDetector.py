import cv2
import numpy as np


class ShapeDetector:

	# detected shapes list
	shapes = []

	# shape names
	_shape_names = [
		None,
		None,
		None,
		"triangle",
		"square",
		"penatgon",
		"hexagon"
	]

	def __init__(self, contours):
		# generate list of detected shapes
		self.shapes = [self._detect_shape(cnt) for cnt in contours]

	# determine circle type
	def _circle_type(self, cnt):
		# calculate closest fitting ellipse and axes lengths (a,b)
		ellipse = cv2.boxPoints(cv2.fitEllipse(cnt))
		a = np.linalg.norm(ellipse[0] - ellipse[1])
		b = np.linalg.norm(ellipse[1] - ellipse[2])

		# calculate ratio of contour area to ellipse area
		area_ratio = (np.pi * a * b) / cv2.contourArea(cnt)
		if np.abs(area_ratio) <= 5:
			return "circle" if np.abs(a - b) <= 10 else "ellipse"
		else:
			return "unknown"

	# determine if quadrilateral has equal sides
	def _sides_equal(self, quad):
		# calculate length of all 4 sides of quadrilateral
		s1 = np.linalg.norm(quad[0] - quad[1]) 
		s2 = np.linalg.norm(quad[1] - quad[2])
		s3 = np.linalg.norm(quad[2] - quad[3])
		s4 = np.linalg.norm(quad[3] - quad[1])
		sides = np.array([s1, s2, s3, s4])

		# return true if all sides are equal, else false 
		return np.mean(np.abs(sides - np.mean(sides))) <= 10

	# determine if quadrilateral has equal diagonols
	def _diag_equal(self, quad):
		# calculate length of both diagonols of quadrilateral
		d1 = np.linalg.norm(quad[0] - quad[2])
		d2 = np.linalg.norm(quad[1] - quad[3])

		# return true if diagonols are equal, else false 
		return np.abs(d1 - d2) <= 10

	# determine quadrilateral type
	def _quad_type(self, poly):
		# determine the quadrilateral type
		if self._sides_equal(poly):
			return "square" if self._diag_equal(poly) else "rhombus"
		else:
			return "rectangle" if self._diag_equal(poly)
				else "parallelogram"
		return "quadrilateral"

	# detect shape
	def _detect_shape(self, cnt):
		# calculate approx. number of sides of closest fitting polygon
		perimeter = cv2.arcLength(cnt, True)
		polygon = cv2.approxPolyDP(cnt, perimeter * 0.01, True)[:,0,:]
		num_sides = polygon.shape[0]

		# determine shape type
		if num_sides == 4:
			return self._quad_type(polygon)
		else:
			return self._shape_names[num_sides]
				if (num_sides < len(self._shape_names))
				else self._circle_type(cnt)

