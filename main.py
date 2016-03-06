import cv2
import numpy as np
import sys

from ColourDetector import ColourDetector
from ShapeDetector import ShapeDetector


WHITE = (255,255,255)


# calculate contour center from moments
def calc_center(cnt):
	M = cv2.moments(cnt)
	cx = np.int0(M["m10"] / M["m00"])
	cy = np.int0(M["m01"] / M["m00"])
	return (cx, cy)


def main():
	print "[ shape and colour classification  ]"

	# get image filname
	img_filename = sys.argv[1] if len(sys.argv) > 1 else "img1.png"

	# load image
	img = cv2.imread("images/%s" % img_filename)
	if img is None:
		print "image could not be loaded!"
		sys.exit()

	# extract shapes
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blur_img = cv2.bilateralFilter(gray_img, 10, 30, 30)
	thresh_img = cv2.threshold(blur_img, 60, 255, cv2.THRESH_BINARY)[1]

	# detect contours
	contours = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]

	# detect colours and shapes
	colours = ColourDetector(contours, img).colours
	shapes = ShapeDetector(contours).shapes

	# draw results
	for cnt, c, s in zip(contours, colours, shapes):
			cx, cy = calc_center(cnt)
			cv2.circle(img, (cx,cy), 3, WHITE, -1)
			cv2.putText(img, c, (cx+5, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
			cv2.putText(img, s, (cx+5, cy+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)

	# display resulting image
	cv2.imwrite("images/result.png", img)
	cv2.imshow("result", img)
	cv2.waitKey(0)


if __name__ == "__main__":
	main()
