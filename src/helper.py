import numpy as np
import cv2
import scipy.ndimage
from numpy.linalg import inv

def q_to_R(q):

    qx,qy,qz,qw = q[0],q[1],q[2],q[3]

    return np.array([
                      [[1 - 2*qy**2 - 2*qz**2],[2*qx*qy - 2*qz*qw],[2*qx*qz + 2*qy*qw]],
                      [[2*qx*qy + 2*qz*qw],[1 - 2*qx**2 - 2*qz**2],[2*qy*qz - 2*qx*qw]],
                      [[2*qx*qz - 2*qy*qw],[2*qy*qz + 2*qx*qw],[1 - 2*qx**2 - 2*qy**2]]
                      ])

def cam_transforms():

    q0 = [0.002,0.001,-0.006,1.000]
    q1 = [0.005,0.002,-0.002,1.000]
    q2 = [-0.000,0.585,-0.010,0.811]
    q3 = [-0.006,0.586,-0.007,0.810]
    q4 = [0.006,0.950,-0.002,0.311]
    q5 = [0.019,0.951,-0.011,0.309]
    q6 = [-0.006,0.951,0.002,-0.310]
    q7 = [-0.012,0.951,0.001,-0.310]
    q8 = [0.004,-0.587,0.005,0.809]
    q9 = [0.002,-0.586,0.008,0.810]

    q = [q0,q1,q2,q3,q4,q5,q6,q7,q8,q9]
    R = []
    for i in q:
        R.append(q_to_R(i))

    return R

def obtain_list(imgIndex):

    # using bottom row images [1,3,5,7,9]
    # centre is 1, [5,3,1,9,7]
    if imgIndex[0] == 1:
        left_list, right_list = [1,3,5], [7,9]

    if imgIndex[0] == 0 and len(imgIndex) == 5:
        left_list, right_list = [0,2,4], [6,8]

    return left_list,right_list

def obtain_index(i):

    # if the current image is taken by the bottom row cam (i is odd)
    if i%2 != 0:
        # the top image is one of the candidate tp = i-1
        # then we take 4 more images from the left and right cameras
        # tpr = i-3 btr = i-2 tpl = i-9 btl = i-8 (cannot use + b/c index might be out of bound)
        return np.array([i-1,i-3,i-2,i-9,i-8])

    #else i is even, the image is taken by the top row cam
    else:
        # the bottom image is one of the candidate bt = i+1
        # then we take 4 more images from the left and right cameras
        # tpr = i-2 btr = i-1 tpl = i-8 btl = i-7
        return np.array([i+1,i-2,i-1,i-8,i-7])

def obtain_good_matches(bf,cur, neighb):

    # use brute force match to find the kth nearest neighbour match
    matches = bf.knnMatch(cur,neighb,k=2)

    # Store all the good matches as per Lowe's ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append([m])

    return good

# obtain homography that warps img (1-9) into coordinates of img 0
def obtain_Hs(H_dict):

    # H_dict{'ba'} warps b into the coordinates of a
    # Already computed H10,H20,H30,H90,H80
    H40 = H_dict[str(4)+str(2)]@H_dict[str(2)+str(0)]
    H60 = H_dict[str(6)+str(8)]@H_dict[str(8)+str(0)]

    Hs = H_dict
    Hs['40'] = H40
    Hs['60'] = H60

    return Hs


def warpTwoImages(img1, img2, H):
    '''warp img2 to img1 with homograph H'''
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    return result
