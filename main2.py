import numpy as np
import cv2 as cv
import scipy as sp
import matplotlib.pyplot as plt
"""
Algorithm: Automatic Panorama Stitching (A simpler version)

Input: n unordered images, in this case n = 10
Output: Panoramic image

I. Extract SIFT features from all n images.
II. Brute-Force Matching with SIFT Descriptors and Ratio Test
III. For each image:
    (i) Select m candidate matching images that have the most feature matches to this images
    (ii) Find consistent feature matches using RANSAC to solve the optimal homography between pairs of images
IV. Find connected components of image matches
V. For each connected component:
    (i) Utilize the cameras parameters from the dataset
    (ii) Render panorama using multi-band blending

"""

# Import all images
img = []
kps = []
dess = []
# initialize sift detector
sift = cv.SIFT_create()
for i in range(10):
    image = cv.imread('./data/omni_image'+str(i)+'/processed.png')
    img.append(image)

    # make current image to grey scale and
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)

    # detect keypoints and compute descriptors from the keypoints
    kp, des = sift.detectAndCompute(gray,None)

    # collect all the keypoints
    kps.append(kp)
    dess.append(des)

# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(dess[4],dess[7],k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img[4],kps[4],img[7],kps[7],good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#plt.imshow(img3),plt.show()

#img=cv.drawKeypoints(gray,kp,img)
cv.imwrite('sift_keypoints.jpg',img3)







# if __name__=="__main__":
