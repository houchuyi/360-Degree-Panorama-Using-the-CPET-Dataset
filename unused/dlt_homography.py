import numpy as np
from numpy.linalg import inv, norm
from scipy.linalg import null_space

def dlt_homography(I1pts, I2pts):
    """
    Find perspective Homography between two images.

    Given 4 points from 2 separate images, compute the perspective homography
    (warp) between these points using the DLT algorithm.

    Parameters:
    -----------
    I1pts  - 2x4 np.array of points from Image 1 (each column is x, y).
    I2pts  - 2x4 np.array of points from Image 2 (in 1-to-1 correspondence).

    Returns:
    --------
    H  - 3x3 np.array of perspective homography (matrix map) between image coordinates.
    A  - 8x9 np.array of DLT matrix used to determine homography.
    """
    #--- FILL ME IN ---

    #obtain all the coordinates from the input arguments
    x,y,u,v = I1pts[0],I1pts[1],I2pts[0],I2pts[1]
    #initialize the 8x9 matrix A
    A = []
    #loop for construting the matrix A (8x9)
    for i in range(0,4):
        xi,yi,ui,vi = x[i],y[i],u[i],v[i]
        A_i1 = [-xi,-yi,-1,0,0,0,ui*xi,ui*yi,ui]
        A_i2 = [0,0,0,-xi,-yi,-1,vi*xi,vi*yi,vi]
        A.append(A_i1)
        A.append(A_i2)

    #convert A into numpy array for further computation of its kernal (i.e [h1 ... h9]^T)
    A = np.array(A)
    #compute null space
    H = null_space(A)
    #normalize the null space vector so that h9 = 1
    H = H/H[8]
    #reshape the null space to 3x3 matrix H
    if H.shape == (9,1):
        H = H.reshape(3,3)
    else:
        return -1,A
    #------------------
    return H, A
