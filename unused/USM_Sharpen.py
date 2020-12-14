import cv2 as cv
import numpy as np

for i in range(10):
    image = cv.imread('./data/omni_image'+str(i)+'/gamma_corrected.png')

    # Perform USM sharpen algorithm
    blur_img = cv.GaussianBlur(image,(0,0),5)
    usm = cv.addWeighted(image,1.4, blur_img,-0.4, 0)
    h, w = image.shape[:2]
    processed = np.zeros([h,w,3],dtype=image.dtype)
    processed[0:h,0:w,:] = usm

    cv.imwrite('./data/omni_image'+str(i)+'/processed.png',processed)

print('Processing Completed.')
