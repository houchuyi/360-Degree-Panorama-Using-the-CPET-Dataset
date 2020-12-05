import numpy as np

def ransac(kp_pairs, tol, iter):
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
    Ipt1, Ipt2 = np.array([]),np.array([]) # points to be extracted to compute H

    for i in range(iter):

        # select 4 random feature pairs

        for j in range(4):
            randpair = (kp_pairs[random.randrange(0,len(pair))])
            Ipt1.hstack((Ipt1,randpair[0].T))
            Ipt2.hstack((Ipt2,randpair[1].T))

        # compute homography
        H = dlt_homography(Ipt1, Ipt2)

        # add inliers
        for pair in feature_pairs:

            kp_1 = np.hstack((pair[:2],1))
            kp_2 = np.hstack((pair[2:4],1))
            H_q = np.dot(H, kp_q)

            # if the distance between the feature 2 and the point transformed from feature 1 by the homography is less than the error tolerance, we add this pair to the inlier set
            if np.linalg.norm(kp_2 - H_q) < tol:
                inlinear.append(pair)

        # keep the largest set of inliers
        if len(inlier) > len(max_inlier):
            max_inlier = inlier
            opt_H = H

    return opt_H, np.array(max_inlier)
