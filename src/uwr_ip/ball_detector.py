'''
Created on 2 Apr 2021

@author: finn
'''
import cv2 as cv
from math import sin, sqrt

class ball_detector( ) :
	'''
	classdocs
	'''

	def __init__(self ) : pass
	'''
	Constructor
	'''

	def custom_sobel(self, shape, axis):
		"""
		shape must be odd: eg. (5,5)
		axis is the direction, with 0 to positive x and 1 to positive y
		"""
		k = np.zeros(shape)
		p = [(j,i) for j in range(shape[0]) 
			for i in range(shape[1]) 
				if not (i == (shape[1] -1)/2. and j == (shape[0] -1)/2.)]
		
		for j, i in p:
			j_ = int(j - (shape[0] -1)/2.)
			i_ = int(i - (shape[1] -1)/2.)
			k[j,i] = (i_ if axis==0 else j_)/float(i_*i_ + j_*j_)
		return k
	
	def custom_circle(self, shape, radius, center = [0,0] ) :
		k = np.zeros( shape )
		for x in range(shape[0]) :
			for y in range(shape[1]) :
				xc= pow( x - center[0], 2 )
				yc= pow( y - center[1], 2 )
				k[x,y] = radius - sqrt( xc+yc )
		k = k / radius
		return k

	def detect_contour( self, frame_ ) :
# 		frame = cv.medianBlur( frame_, 5 )
#		frame = cv.bilateralFilter( frame_, 5, 175, 175 )
		frame = cv.GaussianBlur( frame_, (5 , 5), 0 )
		cv.imshow("frame_lowpassed", frame )
		frame = cv.cvtColor( frame, cv.COLOR_BGR2HSV);
		
# 		frame = cv.normalize(
# 			  frame.astype('float')
# 			, None
# 			, 0.0
# 			, 1.0
# 			, cv.NORM_MINMAX
# 			)

		cv.imshow("h", frame[:,:,0] )

		frame = cv.inRange(
			  frame[:,:,0]
			, 80
			, 90
			)
		
		cv.imshow("yellow", frame )

		contours, hierarchy = cv.findContours(
			  frame
			, cv.RETR_TREE
			, cv.CHAIN_APPROX_SIMPLE
			)
		
		contour_list = []
		for contour in contours:
			approx = cv.approxPolyDP(contour,0.01*cv.arcLength(contour,True),True)
			area = cv.contourArea(contour)
			if ((len(approx) > 8) & (area > 30) ):
				contour_list.append(contour)
		
		cv.drawContours(frame_, contour_list,  -1, (255,0,0), 2)
		cv.imshow('Objects Detected',frame_)
		cv.waitKey(0)
		
		return contour_list
		

	def detect_hough( self, frame_ ) :
		
# 		frame = cv.medianBlur( frame_, 5 )
# 		cv.imshow("frame_blurred", frame )
		frame = cv.cvtColor( frame_, cv.COLOR_BGR2HSV);
# 		cv.imshow("frame_blurred_hsv", frame )
# 		cv.imshow("frame_blurred_h", frame_pre_filter )
		cv.imshow("frame_blurred_h", frame[:,:,0] )
		cv.imshow("frame_blurred_s", frame[:,:,1] )
		cv.imshow("frame_blurred_v", frame[:,:,2] )
# 		return None
		frame_pre_filter = frame[:,:,0];
		cv.imshow("frame_hv.png", frame_pre_filter )
		kshape = (11,11)
		m = self.custom_sobel( kshape , 0 )
		frame_edge = cv.filter2D(
			  frame_pre_filter
			, -1
			, m
			);
		frame_edge = frame_edge + cv.filter2D(
			  frame_pre_filter
			, -1
			, np.fliplr( m )
			);
		m = self.custom_sobel( kshape , 1 )
		frame_edge = frame_edge + cv.filter2D(
			  frame_pre_filter
			, -1
			, m
			);
		frame_edge = frame_edge + cv.filter2D(
			  frame_pre_filter
			, -1
			, np.fliplr( m )
			);
		m = m + self.custom_sobel( kshape, 0 )
		frame_edge = frame_edge + cv.filter2D(
			  frame_pre_filter
			, -1
			, m
			);
		frame_edge = frame_edge + cv.filter2D(
			  frame_pre_filter
			, -1
			, np.flipud( np.fliplr( m ) )
			);

		cv.imshow( "frame_blurred_h_sobel01", frame_edge )
		cv.imwrite("frame_blurred_h_sobel01.png", frame_edge )

		frame_edge = cv.dilate(
			  frame_edge
			, np.ones( (5,5) )
			, 4
			)
		cv.imshow( "frame_blurred_h_sobel01_dilate", frame_edge )
		
		frame_edge = cv.morphologyEx(
			  frame_edge
			, cv.MORPH_GRADIENT
			, np.ones( (5,5) )
			)
		cv.imshow( "frame_blurred_h_sobel01_dilate_morph", frame_edge )

		''' blur image '''
		rows = frame.shape[0]
		''' https://docs.opencv.org/3.4/d4/d70/tutorial_hough_circle.html
		# with the arguments:
		# 
		#     gray: Input image (grayscale).
		#     circles: A vector that stores sets of 3 values: xc,yc,r for each detected circle.
		#     HOUGH_GRADIENT: Define the detection method. Currently this is the only one available in OpenCV.
		#     dp = 1: The inverse ratio of resolution.
		#     min_dist = gray.rows/16: Minimum distance between detected centers.
		#     param_1 = 200: Upper threshold for the internal Canny edge detector.
		#     param_2 = 100*: Threshold for center detection.
		#     min_radius = 0: Minimum radius to be detected. If unknown, put zero as default.
		#     max_radius = 0: Maximum radius to be detected. If unknown, put zero as default
		'''
		circles_vec = cv.HoughCircles(
			  frame_edge
			, cv.HOUGH_GRADIENT
			, 10
			, rows / 8
			, param1 = 100
			, param2 = 30
			, minRadius = 5
			, maxRadius = 30
			)
		
		return circles_vec, frame_edge

if __name__ == "__main__":
	import numpy as np
	frame = cv.imread( "data/GH010751_41237.jpg" )
	
	contours = ball_detector( ).detect_contour( frame )
	
# 	circles, frame_edge = ball_detector( ).detect( frame )
# 	
# 	if circles is not None:
# 		circles = np.uint16(np.around(circles))
# 		for i in circles[0, :]:
# 			center = (i[0], i[1])
# 			# circle center
# 			cv.circle(frame, center, 1, (0, 100, 100), 3)
# 			cv.circle(frame_edge, center, 1, (0, 100, 100), 3)
# 			# circle outline
# 			radius = i[2]
# 			cv.circle(frame, center, radius, (255, 0, 255), 3)
# 			cv.circle(frame_edge, center, radius, (255, 0, 255), 3)
# 	
# 	
# 	cv.imshow("detected circles", frame )
# 	cv.imshow("detected circles", frame_edge )
	cv.waitKey(0)
		