import numpy as np
import cv2 as cv
from support.sift_matchers import *

class Stitch:

    def __init__(self):

        stitch = [cv.imread('./input/omni_image'+str(i)+'/image_for_stitching.png') for i in range(10)]
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
        img_mask = np.copy(img_rgba)
        img_mask[:] = (255,255,255,0)

        return cv.remap(img_rgba, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv.INTER_AREA, borderMode=cv.BORDER_TRANSPARENT), cv.remap(img_mask, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv.INTER_AREA, borderMode=cv.BORDER_TRANSPARENT)

    # Stich two images using a mask. Copy warpedImage in baseImage only if value in mask is 255.
    def Stitch_images(self,baseImage, warpedImage, warpedImageMask):
        rows,cols,_ = baseImage.shape
        for r in range(0,rows):
            for c in range(0,cols):
                if np.max(warpedImageMask[r][c]) == 255:
                    baseImage[r][c] = warpedImage[r][c]
        return baseImage

    def stitching(self):

        print("\n# Part 1: Producing Panorama using both cylindrical warping and perspective warping")
        print("\n# Performing cylindrical warping...")
        print("\n# Please wait... It may take several seconds")
        self.stitchThree([4,6,8],'cylindricalWarp')
        cv.imwrite("./output/4-6-8.jpg", self.pano)
        print('\n# First three images are stitched')

        print("\n# Performing perspective warping for the remaing images...")
        self.stitchThree([2,0],'perspectiveWarp')
        cv.imwrite("./output/2-4-6-8-0-part1.jpg", self.pano)

        y,x,_ = self.pano.shape
        cv.imwrite("./output/2-4-6-8-0-part1-hisEqul.jpg", self.hisEqulColor(self.pano[150:y-200,500:x-300,:]))
        print('\n# Histogram Equalized Cropped Image produced.')

        print("Part 2: Producing Panorama using only perspective warping")
        self.stitchThree([6,4,8,0,2],'perspectiveWarpOnly')
        y,x,_ = self.homopano.shape
        cv.imwrite("./output/2-4-6-8-0-part2.jpg",self.homopano[300:y-100,2100:x-700])

    def stitchThree(self, img_list,warp = 'cylindricalWarp'):

        if warp == 'cylindricalWarp':
            [img2, img1, img3] = [self.images[i] for i in img_list]

            K = [img_list[1],img_list[0],img_list[2]]

            (img1cyl, mask1) = self.cylindricalWarp(img1, self.K[K[0]])
            (img2cyl, mask2) = self.cylindricalWarp(img2, self.K[K[1]])
            (img3cyl, mask3) = self.cylindricalWarp(img3, self.K[K[2]])

            # Add padding to allow space around the center image to paste other images.
            img1cyl = cv.copyMakeBorder(img1cyl,100,100,600,600, cv.BORDER_CONSTANT)

            M21 = self.matcher.feature_matching(img2cyl,img1cyl,"Affine") # match 8 to 6
            M31 = self.matcher.feature_matching(img3cyl,img1cyl,"Affine") # match 4 to 6

            y,x,_ = img1cyl.shape

            transformedImage2 = cv.warpAffine(img2cyl, M21, (x,y))
            transformedMask2 = cv.warpAffine(mask2, M21, (x,y))

            # stitch
            transformedImage21 = self.Stitch_images(img1cyl,transformedImage2,transformedMask2)

            transformedImage3 = cv.warpAffine(img3cyl, M31, (x,y),)
            transformedMask3 = cv.warpAffine(mask3, M31, (x,y))

            # stitch
            transformedImage31 = self.Stitch_images(transformedImage21,transformedImage3,transformedMask3)

            # Use Laplacian_blending to render the panorama
            output = self.Laplacian_blending(transformedImage31,transformedImage21)

            self.pano = output

        if warp == 'perspectiveWarp':

            [img2, img3] = [self.images[i] for i in img_list]

            img1 = cv.imread("./output/4-6-8.jpg")

            # Add padding to the center image to paste other images.
            img1 = cv.copyMakeBorder(img1,0,0,500,500, cv.BORDER_CONSTANT)

            # Calculate homography transformations
            M21 = self.matcher.feature_matching(img2,img1,'Homography') # match 2 to current pano
            M31 = self.matcher.feature_matching(img3,img1,'Homography') # match 0 to current pano

            # Transformed images using homography.
            tranImage2 = cv.warpPerspective(img2, M21, (img1.shape[1],img1.shape[0]), dst=img1.copy(),borderMode=cv.BORDER_TRANSPARENT)

            tranImage3 = cv.warpPerspective(img3, M31, (img1.shape[1],img1.shape[0]), dst=tranImage2.copy(),borderMode=cv.BORDER_TRANSPARENT)

            self.pano = tranImage3

        if warp == 'perspectiveWarpOnly':

            list = [self.images[i] for i in img_list]

            a = list[0]
            i = 0
            for img in list[1:]:
                b = img
                # add border to a to allow images to stitch on it
                a = cv.copyMakeBorder(a,150,150,700,700, cv.BORDER_CONSTANT)
                # find H transfrom from b to a
                try:
                    Mba = self.matcher.feature_matching(b,a,'Homography')

                    # warp images recursively
                    a = cv.warpPerspective(b,Mba, (a.shape[1],a.shape[0]), dst=a.copy(),borderMode = cv.BORDER_TRANSPARENT)

                except (cv.error):
                    print("\nHomography Failed Here, and this could be expected.")
                    print("\nSee output images to spot the potential cause of the failure")
                    continue

            self.homopano = a

            pass

    def Laplacian_blending(self,img1, img2):

        levels = 3
        # generating Gaussian pyramids for both images
        gpImg1 = [img1.astype('float32')]
        gpImg2 = [img2.astype('float32')]
        for i in range(levels):
            img1 = cv.pyrDown(img1)   # Downsampling using Gaussian filter
            gpImg1.append(img1.astype('float32'))
            img2 = cv.pyrDown(img2)
            gpImg2.append(img2.astype('float32'))

        # Generating Laplacin pyramids for both images
        lpImg1 = [gpImg1[levels]]
        lpImg2 = [gpImg2[levels]]

        for i in range(levels,0,-1):
            # Upsampling and subtracting from upper level Gaussian pyramid image to get Laplacin pyramid image
            tmp = cv.pyrUp(gpImg1[i]).astype('float32')
            tmp = cv.resize(tmp, (gpImg1[i-1].shape[1],gpImg1[i-1].shape[0]))
            lpImg1.append(np.subtract(gpImg1[i-1],tmp))

            tmp = cv.pyrUp(gpImg2[i]).astype('float32')
            tmp = cv.resize(tmp, (gpImg2[i-1].shape[1],gpImg2[i-1].shape[0]))
            lpImg2.append(np.subtract(gpImg2[i-1],tmp))

        laplacianList = []
        for lImg1,lImg2 in zip(lpImg1,lpImg2):
            rows,cols,_ = lImg1.shape
            # Merging first and second half of first and second images respectively at each level in pyramid
            mask1 = np.zeros(lImg1.shape)
            mask2 = np.zeros(lImg2.shape)
            mask1[:, 0:cols// 2] = 1
            mask2[:, cols // 2:] = 1

            tmp1 = np.multiply(lImg1, mask1.astype('float32'))
            tmp2 = np.multiply(lImg2, mask2.astype('float32'))
            tmp = np.add(tmp1, tmp2)

            laplacianList.append(tmp)

        img_out = laplacianList[0]
        for i in range(1,levels+1):
            img_out = cv.pyrUp(img_out)   # Upsampling the image and merging with higher resolution level image
            img_out = cv.resize(img_out, (laplacianList[i].shape[1],laplacianList[i].shape[0]))
            img_out = np.add(img_out, laplacianList[i])

        np.clip(img_out, 0, 255, out=img_out)
        return img_out.astype('uint8')


    def hisEqulColor(self,img):
        ycrcb=cv.cvtColor(img,cv.COLOR_BGR2YCR_CB)
        channels=cv.split(ycrcb)
        cv.equalizeHist(channels[0],channels[0])
        cv.merge(channels,ycrcb)
        cv.cvtColor(ycrcb,cv.COLOR_YCR_CB2BGR,img)
        return img

# if __name__ == '__main__':
#
#     s = Stitch()
#     s.stitching()
#     print("\n######################")
#     print ("Done")
#     # s.pano = s.pano[413:703,737:4534]
#     # s.pano = cv.resize(s.pano,(2100,500),interpolation=cv.INTER_CUBIC)
#     cv.destroyAllWindows()
