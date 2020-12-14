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


	def feature_matching(self,img1, img2):
		# Initiate SIFT detector
		sift = cv.SIFT_create()
		# find the keypoints and descriptors with SIFT
		kp1, des1 = sift.detectAndCompute(img1,None)
		kp2, des2 = sift.detectAndCompute(img2,None)
		# FLANN parameters
		FLANN_INDEX_KDTREE = 1
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks=50)   # or pass empty dictionary
		flann = cv.FlannBasedMatcher(index_params,search_params)
		matches2to1 = flann.knnMatch(des2,des1,k=2)

		matchesMask_ratio = [[0,0] for i in range(len(matches2to1))]
		match_dict = {}
		for i,(m,n) in enumerate(matches2to1):
			if m.distance < 0.7*n.distance:
				matchesMask_ratio[i]=[1,0]
				match_dict[m.trainIdx] = m.queryIdx

		# perform reciprocal matching to ensure better matches
		good = []
		recip_matches = flann.knnMatch(des1,des2,k=2)
		matchesMask_ratio_recip = [[0,0] for i in range(len(recip_matches))]

		for i,(m,n) in enumerate(recip_matches):
			if m.distance < 0.7*n.distance: # ratio
				if m.queryIdx in match_dict and match_dict[m.queryIdx] == m.trainIdx:
					good.append(m)
					matchesMask_ratio_recip[i]=[1,0]

		draw_params = dict(matchColor = (0,255,0), singlePointColor = (255,0,0), matchesMask = matchesMask_ratio_recip, flags = 0)
		img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,recip_matches,None,**draw_params)
		cv.imshow("warped", img3)
		cv.waitKey()

		pts1,pts2 =([ kp1[m.queryIdx].pt for m in good ],[ kp2[m.trainIdx].pt for m in good ])

		src_pts = np.float32(pts1).reshape(-1,1,2)
		dst_pts = np.float32(pts2).reshape(-1,1,2)

		M, mask = cv.estimateAffine2D(src_pts, dst_pts, cv.RANSAC, ransacReprojThreshold=5.0)

		return M
