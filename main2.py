import numpy as np
import cv2 as cv
import scipy as sp
from helper import *

"""
Algorithm: Automatic Panorama Stitching (A simpler version)

Input: n unordered images, in this case n = 10
Output: Panoramic image

I. Extract SIFT features from all n images.
II. Brute-Force Matching with SIFT Descriptors and Ratio Test
III. For each image:
    (i) Select m candidate matching images that have the most feature matches to this images (we use m = 5 since we know the omni-camera's configuration and for a given image, there will only be 5 other images that could match it and we know which images are the candidates)
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
    image = cv.imread('./data/omni_image'+str(i)+'/ahe.png')
    img.append(image)

    # make current image to grey scale and
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)

    # detect keypoints and compute descriptors from the keypoints
    kp, des = sift.detectAndCompute(gray,None)

    # collect all the keypoints and descriptors
    kps.append(kp)
    dess.append(des)

bf = cv.BFMatcher()

good = obtain_good_matches(bf,dess[8],dess[0])


img3 = cv.drawMatchesKnn(img[8],kps[8],img[0],kps[0],good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imwrite('Test0-2.jpg',img3)

good = obtain_good_matches(bf,dess[2],dess[4])

img3 = cv.drawMatchesKnn(img[2],kps[2],img[4],kps[4],good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imwrite('Test2-4.jpg',img3)

# if __name__=="__main__":
