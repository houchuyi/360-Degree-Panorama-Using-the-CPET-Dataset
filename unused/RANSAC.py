import numpy as np
from dlt_homography import *
def RANSAC(kp_pairs, tol, iter):
    """
    RANSAC Alogrithm

    RANSAC is essentially a sampling approach to estimating H, the homography.
    In the case of panoramas we select sets of r = 4 feature correspondences and compute the homography H.

    We repeat this with iter = 500 trials and select the solution that has the maximum number
    of inliers

    Parameters:
    -----------
    kp_pairs - np.array of feature pairs founded
    tol   - float error tolerance
    iter  - int number of iterations

    Returns:
    --------
    max_in  -  np.array, largest set of inliers.

    opt_H   -  optimal H

    """
    # initializations
    inlier = [] # contain inliers
    max_inlier = [] # contain the largest set of inliers
    opt_H = None # optimal homography
    Ipt1, Ipt2 = np.zeros([2,4]),np.zeros([2,4]) # points to be extracted to compute H

    for i in range(iter):

        # select 4 random feature pairs

        for j in range(4):
            randpair = (kp_pairs[np.random.random_integers(0,len(kp_pairs)-1)])
            Ipt1[0,j], Ipt1[1,j] = randpair[0], randpair[1]
            Ipt2[0,j], Ipt2[1,j] = randpair[2], randpair[3]

        # compute homography
        H,A = dlt_homography(Ipt2, Ipt1)

        # add inliers
        for pair in kp_pairs:

            kp_1 = np.hstack((pair[:2],1))
            kp_2 = np.hstack((pair[2:4],1))
            H_q = np.dot(H, kp_2)

            # if the distance between the feature 2 and the point transformed from feature 1 by the homography is less than the error tolerance, we add this pair to the inlier set
            if np.linalg.norm(kp_1 - H_q) < tol:
                inlier.append(pair)

        # keep the largest set of inliers
        if len(inlier) > len(max_inlier):
            max_inlier = inlier
            opt_H = H

    return opt_H
