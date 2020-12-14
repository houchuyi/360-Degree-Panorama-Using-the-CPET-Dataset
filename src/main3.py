import numpy as np
import cv2 as cv
import sys
from sift_matchers import *
import time
from matplotlib.path import Path
import math

class Stitch:

    def __init__(self):

        stitch = [cv.imread('../data/omni_image'+str(i)+'/image_for_stitching.png') for i in range(10)]
        self.images = [img for img in stitch]
        self.matcher = sift_matcher()

        # camera intrinsics known
        K0 = np.array([[482.047,0,373.237],[0,485.211,211.02],[0,0,1]])
        K1 = np.array([[479.429,0,367.111],[0,482.666,230.626],[0,0,1]])
        K2 = np.array([[483.259,0,340.948],[0,486.027,204.701],[0,0,1]])
        K3 = np.array([[483.895,0,375.161],[0,486.584,220.184],[0,0,1]])
        K4 = np.array([[473.571,0,378.17],[0,477.53,212.577],[0,0,1]])
        K5 = np.array([[473.368,0,371.65],[0,477.558,204.79],[0,0,1]])
        K6 = np.array([[476.784,0,381.798],[0,479.991,205.64],[0,0,1]])
        K7 = np.array([[480.086,0,361.268],[0,483.581,221.179],[0,0,1]])
        K8 = np.array([[478.614,0,377.363],[0,481.574,194.839],[0,0,1]])
        K9 = np.array([[480.918,0,386.897],[0,484.086,206.923],[0,0,1]])
        self.K = [K0,K1,K2,K3,K4,K5,K6,K7,K8,K9]

    def cylindricalWarp(self,img, K):
        """This function returns the cylindrical warp for a given image and intrinsics matrix K"""
        h_,w_ = img.shape[:2]
        # pixel coordinates
        y_i, x_i = np.indices((h_,w_))
        X = np.stack([x_i,y_i,np.ones_like(x_i)],axis=-1).reshape(h_*w_,3) # to homog
        Kinv = np.linalg.inv(K)
        X = Kinv.dot(X.T).T # normalized coords
        # calculate cylindrical coords (sin\theta, h, cos\theta)
        A = np.stack([np.sin(X[:,0]),X[:,1],np.cos(X[:,0])],axis=-1).reshape(w_*h_,3)
        B = K.dot(A.T).T # project back to image-pixels plane
        # back from homog coords
        B = B[:,:-1] / B[:,[-1]]
        # make sure warp coords only within image bounds
        B[(B[:,0] < 0) | (B[:,0] >= w_) | (B[:,1] < 0) | (B[:,1] >= h_)] = -1
        B = B.reshape(h_,w_,-1)

        img_rgba = cv.cvtColor(img,cv.COLOR_BGR2BGRA) # for transparent borders...
        # warp the image according to cylindrical coords
        return cv.remap(img_rgba, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv.INTER_AREA, borderMode=cv.BORDER_TRANSPARENT)


    def stitchThree(self):

        limg,img2, img1, img3, rimg = self.images[2],self.images[4],self.images[6],self.images[8],self.images[0]

        m = np.ones_like(img3, dtype='float32') # 8
        m1 = np.ones_like(img2, dtype='float32') # 4
        mL = np.ones_like(limg, dtype='float32') # 2
        mR = np.ones_like(rimg, dtype='float32') # 0

        input = [m,m1,mL,mR,img1, img2, img3,limg, rimg]
        K = [8,4,2,0,6,4,8,2,0]
        for i in range(len(input)):
            h,w,_ = input[i].shape
            input[i] = self.cylindricalWarp(input[i],self.K[K[i]])
            #input[i] = cv.copyMakeBorder(input[i],100,100,1200,1200, cv.BORDER_CONSTANT)

        M = self.matcher.feature_matching(input[-3],input[4]) # match 8 to 6
        M1 = self.matcher.feature_matching(input[-4],input[4]) # match 4 to 6
        ML = self.matcher.feature_matching(input[-2],input[-3]) # match 0 to 8
        MR = self.matcher.feature_matching(input[-1],input[-4]) # match 2 to 4


        y,x,_ = input[4].shape

        out1img = cv.warpAffine(input[-3], M, (x,y))
        out2img = cv.warpAffine(input[-4], M1, (x,y))
        out1mask = cv.warpAffine(input[2], M, (x,y))
        out2mask = cv.warpAffine(input[3], M1, (x,y))

        outLimg = cv.warpAffine(input[2], ML@M, (x,y))
        outRimg = cv.warpAffine(input[3], MR@M1, (x,y))
        outLmask = cv.warpAffine(input[2], ML@M, (x,y))
        outRmask = cv.warpAffine(input[3], MR@M1, (x,y))


        # Use Laplacian_blending to render the panorama
        lpb = self.Laplacian_blending(out1img,input[4],out1mask,3)
        lpb1 = self.Laplacian_blending(out2img,lpb,out2mask,3)

        lpb2 = self.Laplacian_blending(outRimg,lpb1,outRmask,3)
        lpb3 = self.Laplacian_blending(outLimg,lpb2,outLimg,3)

        self.pano = lpb3

    def fullPano(self):

        self.pano = cv.imread('../stitchThree.jpg')
        img1, img2, img3 = self.images[0],self.pano,self.images[2]

        m = np.ones_like(img3, dtype='float32')
        m1 = np.ones_like(img2, dtype='float32')

        input = [m,m1,img1, img2, img3]
        K = [0,2,6,2,0]
        for i in range(len(input)):
            if i == 3:
                continue
            else:
                h,w,_ = input[i].shape
                input[i] = self.cylindricalWarp(input[i],self.K[K[i]])
                input[i] = cv.copyMakeBorder(input[i],100,100,600,600, cv.BORDER_CONSTANT)

            cv.imshow("here", input[i])
            cv.waitKey()

        M = self.matcher.feature_matching(input[-1],input[2])
        M1 = self.matcher.feature_matching(input[-2],input[2])

        y,x,_ = input[2].shape

        out1 = cv.warpAffine(input[-1], M, (x,y))
        out2 = cv.warpAffine(input[-2], M1, (x,y))
        out3 = cv.warpAffine(input[0], M, (x,y))
        out4 = cv.warpAffine(input[1], M1, (x,y))

        cv.imshow("warped", out1)
        cv.waitKey()
        cv.imshow("warped", out2)
        cv.waitKey()

        # Use Laplacian_blending to render the panorama
        lpb = self.Laplacian_blending(out1,input[2],out3,4)
        lpb1 = self.Laplacian_blending(out2,lpb,out4,4)

        self.pano = lpb1

    def Laplacian_blending(self,img1,img2,mask,levels=4):

        G1 = img1.copy()
        G2 = img2.copy()
        GM = mask.copy()
        gp1 = [G1]
        gp2 = [G2]
        gpM = [GM]
        for i in range(levels):
            G1 = cv.pyrDown(G1)
            G2 = cv.pyrDown(G2)
            GM = cv.pyrDown(GM)
            gp1.append(np.float32(G1))
            gp2.append(np.float32(G2))
            gpM.append(np.float32(GM))

        # generate Laplacian Pyramids for A,B and masks
        lp1  = [gp1[levels-1]] # the bottom of the Lap-pyr holds the last (smallest) Gauss level
        lp2  = [gp2[levels-1]]
        gpMr = [gpM[levels-1]]
        for i in range(levels-1,0,-1):
            # Laplacian: subtarct upscaled version of lower level from current level
            # to get the high frequencies
            L1 = np.subtract(gp1[i-1], cv.pyrUp(gp1[i]))
            L2 = np.subtract(gp2[i-1], cv.pyrUp(gp2[i]))
            lp1.append(L1)
            lp2.append(L2)
            gpMr.append(gpM[i-1]) # also reverse the masks

        # Now blend images according to mask in each level
        LS = []
        for l1,l2,gm in zip(lp1,lp2,gpMr):
            ls = l1 * gm + l2 * (1.0 - gm)
            LS.append(ls)

        # now reconstruct
        ls_ = LS[0]
        for i in range(1,levels):
            ls_ = cv.pyrUp(ls_)
            ls_ = cv.add(ls_, LS[i])

        return ls_

if __name__ == '__main__':

    s = Stitch()
    s.stitchThree()
    cv.imwrite("../stitchThree.jpg", s.pano)
    # s.fullPano()
    # cv.imwrite("../fullPano.jpg", s.pano)
    print ("done")
    # s.pano = s.pano[413:703,737:4534]
    # s.pano = cv.resize(s.pano,(2100,500),interpolation=cv.INTER_CUBIC)
    cv.imwrite("../testCyl.jpg", s.pano)
    print ("image written")
    cv.destroyAllWindows()
