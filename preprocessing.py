import cv2 as cv
import numpy as np
import shutil

src = 'C:/Users/skyho/Desktop/YEAR 4/ROB501/Assignments/Project/data/omni_image'
dst = './data/omni_image'
frame = '/frame000592_2018_09_04_17_45_39_905367.png'

for i in range(0,10):
    shutil.copy2(src+str(i)+frame, dst+str(i)+'/image.png') # complete target filename given

print("Copy images completed")

def gamma_trans(img,gamma):#gamma函数处理
    gamma_table=[np.power(x/255.0,gamma)*255.0 for x in range(256)]#建立映射表
    gamma_table=np.round(np.array(gamma_table)).astype(np.uint8)#颜色值为整数
    return cv.LUT(img,gamma_table)#图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。

def get_Gamma_Value(gray_img):
    mean_value = cv.mean(gray_img)

    val = mean_value[0]
    gamma_val = (np.log10(0.5))/(np.log10(val / 255.0))

    return gamma_val

for i in range(10):

    # if i == 2:
    #     image = cv.imread('./data/omni_image'+str(i)+'/image.png')
    # else:
    # image = cv.imread('./data/omni_image'+str(i)+'/calibresult.png')

    ## Gamma Correction
    img = cv.imread('./data/omni_image'+str(i)+'/image.png')
    gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    value_of_gamma= get_Gamma_Value(gray_img)#gamma取值
    # value_of_gamma=value_of_gamma*0.01#压缩gamma范围，以进行精细调整
    image_gamma_correct=gamma_trans(img,value_of_gamma)#2.5为gamma函数的指数值，大于1曝光度下降，大于0小于1曝光度增强
    h, w = img.shape[:2]
    cv.imwrite('./data/omni_image'+str(i)+'/gamma_corrected.png',image_gamma_correct[3:h-3,0:w,:])

    ## Histogram equalization
    equ = cv.equalizeHist(gray_img)
    res = np.hstack((gray_img,equ)) #stacking images side-by-side
    cv.imwrite('./data/omni_image'+str(i)+'/he.png',res[3:h-3,0:w])

    ## adaptive histogram equalization
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(gray_img)
    cv.imwrite('./data/omni_image'+str(i)+'/ahe.png',cl1[3:h-3,0:w])

print('Processing Completed.')
