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

    return [K0,K1,K2,K3,K4,K5,K6,K7,K8,K9],[d0,d1,d2,d3,d4,d5,d6,d7,d8,d9]

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

def obtain_index(i):

    # if the current image is taken by the bottom row cam (i is odd)
    if i%2 != 0:
        # the top image is one of the candidate tp = i-1
        # then we take 4 more images from the left and right cameras
        # tpr = i-3 btr = i-2 tpl = i-9 btl = i-8 (cannot use + b/c index might be out of bound)
        # return np.array([i-3,i-9])
        return np.array([i-1,i-3,i-2,i-9,i-8])

    #else i is even, the image is taken by the top row cam
    else:
        # the bottom image is one of the candidate bt = i+1
        # then we take 4 more images from the left and right cameras
        # tpr = i-2 btr = i-1 tpl = i-8 btl = i-7
        return np.array([i+1,i-2,i-1,i-8,i-7])
        # return np.array([i-2,i-8])
def obtain_good_matches(bf,cur, neighb):

    # use brute force match to find the kth nearest neighbour match
    matches = bf.knnMatch(cur,neighb,k=2)

    # Store all the good matches as per Lowe's ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append([m])

    return good

def obtain_H_from_T():
    # H=[[a1 a2 a3] [a4 a5 a6] [a7 a8 a9]]
    # R= [[a1 a2] [a4 a5]], T=[[a3] [a6]]
    pass

# obtain homography that warps img (1-9) into coordinates of img 0
def obtain_Hs(H_dict):

    # H_dict{'ba'} warps b into the coordinates of a
    # Already computed H10,H20,H30,H90,H80
    H40 = H_dict[str(4)+str(2)]@H_dict[str(2)+str(0)]
    H60 = H_dict[str(6)+str(8)]@H_dict[str(8)+str(0)]
    Hs = [H_dict[str(2)+str(0)],
          H40,#/H40[2,2],
          H60,#/H60[2,2],
          H_dict[str(8)+str(0)]]
        # error may accumulate here

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

# used to crop stitched image at the end
def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop bottom
    elif not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop left
    elif not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop right
    elif not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame


def panorama(H,fromim,toim,padding=2400,delta=2400):
    #使用单应性矩阵H，协调两幅图片，创建水平全景图像，结果为一幅和toim具有相同高度的图像。padding指定填充像素的数目，delta指定额外的平移量。

    def transf(p):
        p2 = np.dot(H,[p[0],p[1],1])
        return (p2[0]/p2[2],p2[1]/p2[2])

    if H[1,2]<0: #fromin在右边
        print ('warp - right')

        #在目标图像的右边填充0
        toim_t = np.hstack((toim,np.zeros((toim.shape[0],padding,3))))
        fromim_t = np.zeros((toim.shape[0],toim.shape[1]+padding,toim.shape[2]))
        for col in range(3):
            fromim_t[:,:,col] = scipy.ndimage.geometric_transform(fromim[:,:,col], transf,(toim.shape[0],toim.shape[1]+padding))

    else:#fromin在左边
        print ('warp - left')

        #为了填充补偿效果，在左边加入平移量
        H_delta = np.array([[1,0,0],[0,1,-delta],[0,0,1]])
        H = np.dot(H,H_delta)
        # 在目标图像左边填充0的代码与上述类似，便不再列出。
        toim_t = np.hstack((np.zeros((toim.shape[0],padding,3)),toim))
        fromim_t = np.zeros((toim.shape[0],toim.shape[1]+padding,toim.shape[2]))
        for col in range(3):
            fromim_t[:,:,col] = scipy.ndimage.geometric_transform(fromim[:,:,col], transf,(toim.shape[0],toim.shape[1]+padding))


    # merge images using alpha blending
    alpha = ((fromim_t[:,:,0] * fromim_t[:,:,1] * fromim_t[:,:,2] ) > 0)
    for col in range(3):
        toim_t[:,:,col] = fromim_t[:,:,col]*alpha + toim_t[:,:,col]*(1-alpha)

    return toim_t
