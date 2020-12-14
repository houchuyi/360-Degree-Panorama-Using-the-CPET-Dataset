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

# I. initialize sift detector
sift = cv.SIFT_create()
for i in [0,2,4,6,8]:

    # import images for stitching
    image_for_stitching = cv.imread('./data/omni_image'+str(i)+'/processed.png')
    stimg[str(i)] = image_for_stitching

    # import processed images for matching purpose
    image_for_matching = cv.imread('./data/omni_image'+str(i)+'/ahe.png')
    img[str(i)]= image_for_matching

    # # make current image to grey scale and
    # gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)

    # detect keypoints and compute descriptors from the keypoints
    kp, des = sift.detectAndCompute(image_for_matching,None)

    # collect all the keypoints and descriptors
    kps[str(i)]= kp
    dess[str(i)]= des

# III. For each image
# initialize the brute force matcher
bf = cv.BFMatcher()
for cur_index in [0,2,4,6,8]:
    # (i) we know the 5 candidate matching images because they are taken by the 5 neighbour cameras
    # cand_images = []

    # obtain the image index, i.e. the camera index for the 5 neighbour images relative to the current image index
    index = obtain_index(cur_index)
    index = np.where(index<0, index+10, index)
    for neighb_index in index:

        if str(neighb_index) not in img:
            continue

        else:
            # cand_images.append(img[str(neighb_index)])

            # II. Use brute force matching, BFMatcher with default params
            good = obtain_good_matches(bf,dess[str(cur_index)],dess[str(neighb_index)])

            # obtain pixel location pairs from the good matches
            # ls = []
            # for match in good:
            #     l1 = list(kps[str(cur_index)][match.queryIdx].pt)
            #     l1.extend(list(kps[str(neighb_index)][match.trainIdx].pt))
            #     ls.append(l1)
            # ls = np.array(ls)

            # (ii)
            # for each pair of images (cur_image, neighb_image), using RANSAC to filter inliers of the good matches and to solve for the homography
            print(cur_index,neighb_index)
            # if cur_index == 0 and neighb_index == 2:
            #
            #     img3 = cv.drawMatchesKnn(img[str(cur_index)],kps[str(cur_index)],img[str(neighb_index)],kps[str(neighb_index)],good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            #     cv.imwrite('BFtest.jpg',img3)

            if (len(good) >= 4):
                dst = np.float32([ kps[str(cur_index)][m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
                src = np.float32([ kps[str(neighb_index)][m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)

            H, masked = cv.findHomography(src, dst, cv.RANSAC, 5.0)
            # H = RANSAC(ls, 3, 200)

            H_dict[str(neighb_index)+str(cur_index)] = H

# obtain homography that warps img (1-9) into coordinates of img 1
Hs = obtain_Hs(H_dict)

# warp image
h = img['0'].shape[0]
w = img['0'].shape[1]
wrap = np.zeros([h,w,3],dtype=stimg['0'].dtype)

for i in range(9):
    print(i)
    if str(i+1) not in stimg:
        continue
    else:
        wrap = warpTwoImages(stimg['0'],stimg[str(i+1)], Hs[i//2])
        # wrap = panorama(Hs[i],img[i+1],img[0])
        cv.imwrite('firststitch'+str(0)+'-'+str(i+1)+'.jpg',wrap)

# wrap = warpTwoImages(img[0],img[4],H_dict[str(8)+str(0)])
# cv.imwrite('firststitch.jpg',wrap)


    # print(cur_index,neighb_index)
    # print(len(good))
    # print(good)
    # if cur_index == 4 and neighb_index == -3:
    #     img3 = cv.drawMatchesKnn(img[4],kps[4],img[-3],kps[-3],good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #     cv.imwrite('BFtest.jpg',img3)


# if __name__=="__main__":
