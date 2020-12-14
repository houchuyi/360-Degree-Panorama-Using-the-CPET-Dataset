import cv2 as cv
import numpy as np
from helper import *

# obtain camera intrinscs from our dataset
mtx, dist = cam_intrinsics()

for i in range(10):

    img = cv.imread('./data/omni_image'+str(i)+'/image.png')
    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx[i], dist[i], (w,h), 1, (w,h))

    # undistort
    dst = cv.undistort(img, mtx[i], dist[i], None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    res=cv.resize(dst,(752,480),interpolation=cv.INTER_CUBIC)

    cv.imwrite('./data/omni_image'+str(i)+'/calibresult.png', res)
