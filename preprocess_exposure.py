import cv2 as cv
import numpy as np

def main():

    for i in range(10):
        image = cv.imread('./data/omni_image'+str(i)+'/image.png')

        equalized = np.zeros(image.shape, image.dtype)
        equalized[:, :, 0] = cv.equalizeHist(image[:, :, 0])
        equalized[:, :, 1] = cv.equalizeHist(image[:, :, 1])
        equalized[:, :, 2] = cv.equalizeHist(image[:, :, 2])
        cv.imwrite('./data/omni_image'+str(i)+'/equlized.png', equalized)

    print('Processing Completed.')

if __name__ == '__main__':
    main()
