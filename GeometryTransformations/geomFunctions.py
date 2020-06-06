import numpy as np 
import cv2



def vec2homo(tvec, rvec):
    R, _ = cv2.Rodrigues(rvec)
    H = np.eye(4)
    H[:3,:3] = R
    H[:3, 3] = tvec
    return H


def homoInv(H):
    # prepare 
    invH = np.eye(4)
    tvec = H[:3, 3]
    R = H[:3,:3]
    # compute
    invH[:3,:3] = np.transpose(R)
    invH[:3, 3] = -np.dot(np.transpose(R), tvec)
    return invH






























