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
    (i) Select m candidate matching images that have the most feature matches to this images (we use m = 5 since we know the omni-camera's configuration and for a given image, there will only be 5 other images that could match it)
    (ii) Find consistent feature matches using RANSAC to solve the optimal homography between pairs of images
IV. Find connected components of image matches
V. For each connected component:
    (i) Utilize the cameras parameters from the dataset
    (ii) Render panorama using multi-band blending

"""

# initializations
img = []
kps = []
dess = []

m = 5 # number of candidate matching images

# I. initialize sift detector
sift = cv.SIFT_create()
for i in range(10):

    # import images
    image = cv.imread('./data/omni_image'+str(i)+'/image.png')
    img.append(image)

    # make current image to grey scale and
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)

    # detect keypoints and compute descriptors from the keypoints
    kp, des = sift.detectAndCompute(gray,None)

    # collect all the keypoints
    kps.append(kp)
    dess.append(des)

# II. Use brute force matching, BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(dess[4],dess[5],k=2)

# Store all the good matches as per Lowe's ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# III. For each image
for image in img:

    # (i) select 5 candidate matching images that have the most feature matches to this iamge
    for i in range(m):






# if __name__=="__main__":
