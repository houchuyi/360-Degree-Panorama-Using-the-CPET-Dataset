'''
This file is only for importing images.
For the project, you do not need to run this file since the they are already inside input folder
If not, you may want to change the src directory and import the images accordingly.
'''

import cv2 as cv
import numpy as np
import shutil

src = 'C:/Users/skyho/Desktop/YEAR 4/ROB501/Assignments/Project/data/omni_image'
dst = '../data/omni_image'
input = '../../input/omni_image'
frame = '/frame000187_2018_09_04_17_44_08_405404.png' #frame000187_2018_09_04_17_44_08_405404.png

for i in range(0,10):
    shutil.copy2(src+str(i)+frame, input+str(i)+'/image.png')

print("Copy images completed")

def gamma_trans(img,gamma):
    gamma_table=[np.power(x/255.0,gamma)*255.0 for x in range(256)]#create lookup table
    gamma_table=np.round(np.array(gamma_table)).astype(np.uint8)#intensity is int
    return cv.LUT(img,gamma_table)#image colour lookup table

def get_Gamma_Value(gray_img):
    mean_value = cv.mean(gray_img)

    val = mean_value[0]
    gamma_val = (np.log10(0.5))/(np.log10(val / 255.0))

    return gamma_val

def cam_intrinsics():

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

    d0 = np.array([-0.332506,0.154213,-9.5973e-05,-0.000236179,-0.0416498])
    d1 = np.array([-0.334792,0.161382,4.29188e-05,-0.000324466,-0.0476611])

    d3 = np.array([-0.337111,0.160611,0.000146382,0.000406074,-0.0464726])
    d4 = np.array([-0.333605,0.159377,6.11251e-05,4.90177e-05,-0.0460505])
    d5 = np.array([-0.3355,0.162877,4.34759e-05,2.72184e-05,-0.0472616])
    d6 = np.array([-0.334747,0.162797,-0.000305541,0.000163014,-0.0517717])
    d7 = np.array([-0.348515,0.199388,-0.000381909,8.83314e-05,-0.0801161])
    d8 = np.array([-0.333512,0.157163,-8.2852e-06,0.000265461,-0.0447446])
    d9 = np.array([-0.33305,0.156207,-5.95668e-05,0.000376887,-0.0438085])

    return [K0,K1,K1,K3,K4,K5,K6,K7,K8,K9],[d0,d1,d1,d3,d4,d5,d6,d7,d8,d9]

mtx, dist = cam_intrinsics()

for i in range(10):

    img = cv.imread(input+str(i)+'/image.png')
    h,  w = img.shape[:2]
    ###################################################################################
    # undistort
    # newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx[i], dist[i], (w,h), 1, (w,h))
    # dst = cv.undistort(img, mtx[i], dist[i], None, newcameramtx)
    # # crop the image
    # x, y, w, h = roi
    # dst = dst[y:y+h, x:x+w]
    # img=cv.resize(dst,(752,480),interpolation=cv.INTER_CUBIC)
    ###################################################################################

    ###################################################################################
    ## Gamma Correction
    gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    value_of_gamma= get_Gamma_Value(gray_img)*0.9#get gamma value
    image_gamma_correct=gamma_trans(img,value_of_gamma)#>1 decrease exposure,0<x<1 increase
    h, w = img.shape[:2]
    cv.imwrite(input+str(i)+'/image_for_stitching.png',image_gamma_correct[0:h,0:w,:])
    ###################################################################################


    ## Histogram equalization
    # equ = cv.equalizeHist(gray_img)
    # res = np.hstack((gray_img,equ)) #stacking images side-by-side
    # cv.imwrite('./data/omni_image'+str(i)+'/he.png',res[3:h-3,0:w])

    ###################################################################################
    ## adaptive histogram equalization
    # clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # cl1 = clahe.apply(gray_img)
    # cv.imwrite('../data/omni_image'+str(i)+'/image_for_matching.png',cl1[3:h-3,2:w-2])
    ###################################################################################


print('Processing Completed.')
