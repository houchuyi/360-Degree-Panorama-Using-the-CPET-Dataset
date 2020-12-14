import cv2 as cv
import numpy as np

'''
Create a feature matching class Using OpenCV SIFT Detector and Brute Force Matching
'''

class sift_matcher:
	def __init__(self):
		self.sift = cv.SIFT_create()
		self.bf = cv.BFMatcher()

	# use adaptive_histogram_equilization to make the matching better
	def adaptive_histogram_equilization(self,img):
		## adaptive histogram equalization
		h, w = img.shape[:2]
		gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
		clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		cl1 = clahe.apply(gray_img)
		return cl1#[3:h-3,0:w]

	def match(self, I1, I2):
		FSet1 = self.getFeatures(I1)
		FSet2 = self.getFeatures(I2)
		matches = self.bf.knnMatch(FSet2['des'],FSet1['des'],k=2)
		good = []
		good1 = []
		for m, n in matches:
			if m.distance < 0.7*n.distance:
				good.append((m.trainIdx, m.queryIdx))
				good1.append([m])
		img3=cv.drawMatchesKnn(I2,FSet2['kp'],I1,FSet1['kp'],good1,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
		cv.imshow("warped", img3)
		cv.waitKey()

		if len(good) > 4:
			cur = FSet2['kp']
			pre = FSet1['kp']

			matchedCur = np.float32(
				[cur[i].pt for (__, i) in good]
			)
			matchedPre = np.float32(
				[pre[i].pt for (i, __) in good]
				)

			# H, _ = cv.findHomography(matchedCur, matchedPre, cv.RANSAC, 5)
			H, mask = cv.estimateAffine2D(matchedCur, matchedPre, cv.RANSAC, ransacReprojThreshold=5.0)
			return H
		return None

	def getFeatures(self, img):
		img = self.adaptive_histogram_equilization(img)
		kp, des = self.sift.detectAndCompute(img, None)
		return {'kp':kp, 'des':des}
