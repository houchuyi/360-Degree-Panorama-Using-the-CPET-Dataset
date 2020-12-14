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
stimg = {}
img = {}
kps = {}
dess = {}
H_dict = {}

# take images
upper_row = [0,2,4,6,8]
bottom_row = [1,3,5,7,9]
All = [0,1,2,3,4,5,6,7,8,9]

# set current images to be stitched
imgIndex = upper_row

# obtain stitching order list
left_list, right_list = obtain_list(imgIndex)

# I. initialize sift detector
sift = cv.SIFT_create()
for i in imgIndex:

    # import images for stitching
    image_for_stitching = cv.imread('./data/omni_image'+str(i)+'/image_for_stitching.png')
    stimg[str(i)] = image_for_stitching

    # import processed images for matching purpose
    image_for_matching = cv.imread('./data/omni_image'+str(i)+'/image_for_matching.png')
    img[str(i)]= image_for_matching

    # detect keypoints and compute descriptors from the keypoints
    kp, des = sift.detectAndCompute(image_for_matching,None)

    # collect all the keypoints and descriptors
    kps[str(i)]= kp
    dess[str(i)]= des

# III. For each image
# initialize the brute force matcher
bf = cv.BFMatcher()
for cur_index in imgIndex:
    # (i) we know the 5 candidate matching images because they are taken by the 5 neighbour cameras

    # obtain the image index, i.e. the camera index for the 5 neighbour images relative to the current image index
    index = obtain_index(cur_index)
    index = np.where(index<0, index+10, index)
    for neighb_index in index:

        if str(neighb_index) not in stimg:
            continue

        else:

            # II. Use brute force matching, BFMatcher with default params
            good = obtain_good_matches(bf,dess[str(cur_index)],dess[str(neighb_index)])

            # (ii)
            # for each pair of images (cur_image, neighb_image), using RANSAC to filter inliers of the good matches and to solve for the homography
            print(cur_index,neighb_index)

            if (len(good) >= 4):
                dst = np.float32([ kps[str(cur_index)][m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
                src = np.float32([ kps[str(neighb_index)][m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)

            H, masked = cv.findHomography(src, dst, cv.RANSAC, 4)
            H_dict[str(neighb_index)+str(cur_index)] = H

# obtain homography that warps img (1-9) into coordinates of img 1
Hs = obtain_Hs(H_dict)
# warp image
#wrap  = warpTwoImages(stimg['0'],stimg[str(2)], H_dict[str(2)+str(0)])


leftImage = leftshift(stimg,left_list,Hs)
# for i in range(9):
#     if str(i+1) not in stimg:
#         continue
#     else:
#         wrap  = warpTwoImages(stimg['0'],stimg[str(i+1)], Hs[i//2])
#         cv.imwrite('firststitch'+str(0)+'-'+str(i+1)+'.jpg',wrap)
