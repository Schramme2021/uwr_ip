'''
Created on 2 Apr 2021

@author: finn
'''
from uwr_ip import ball_detector
import unittest
import cv2 as cv


class Test(unittest.TestCase):
	detector = ball_detector.ball_detector( )
	
	def setUp(self):
		self.__frame = cv.imread( "data/GH010751_41237.jpg" )
	
	
	def tearDown(self) : pass
	
	
	def testName(self) : pass
	
	def test_simple_detection( self ) :
		circles_vec = Test.detector.detect( self.__frame )


if __name__ == "__main__":
	#import sys;sys.argv = ['', 'Test.testName']
	unittest.main( )