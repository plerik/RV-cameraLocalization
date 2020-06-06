import numpy as np 
import cv2
import matplotlib.pyplot as plt

# Functions that we need for application


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

def homo2Rt(H):
    tvec = H[:3, 3]
    R = H[:3,:3]
    return R, tvec.reshape((3,1))


def id2homo(id, ids, tvecs, rvecs):
    '''
        computes marker's homo pose from id of marker (if the marker id is in the ids)
        and tvecs and rvecs.
        if no id is in the ids returns 0
    '''
    wantedMarkerId = id # Id that we want to take a look at
    wantedIdxs = np.nonzero(ids == wantedMarkerId)
    wantedIdxs = wantedIdxs[0] # Because array ids is actually 1D

    if wantedIdxs.shape[0] > 0:
        if  wantedIdxs.shape[0] > 1:
            print('Warning! More than one marker with id {wantedMarkerId}.')
            print('The program will use one of them')
        wantedIdx = wantedIdxs[0]

        H = vec2homo(tvecs[wantedIdx], rvecs[wantedIdx])
        return 1, H
    else: 
        return 0, 0
    

# ####################################################################################
# ### Odometry and Pose reconstruction related 
# ####################################################################################

def points3Dreconstruction(p1, p2, Re, te, cameraMatrix):
    '''
        Reconstruct 3D points (in homogenous form Nx4) from two camera images knowing 
        their relative position. 

        - p1, p2 sets of corresponding points (Nx2) from 2 consecutive camera images
        - Re, te estimated rotation matrix and translation vector (camera 2 in 
        relation to camera 1)
        - cameraMatrix - camera matrix for the camera (it's supposed that both photos
        are taken with the same camera.)
    
    '''
    R1 = np.eye(3)
    tvec1 = np.array([[0., 0, 0]]).T
    M1 = np.dot(cameraMatrix, np.hstack((R1, tvec1)))
    M2 = np.dot(cameraMatrix, np.hstack((Re, te)))

    points3Dreconstructed = cv2.triangulatePoints(M1, M2, p1.T, p2.T)
#    
    points3Dreconstructed /= points3Dreconstructed[3]
    points3Dreconstructed = np.squeeze(points3Dreconstructed)
    return points3Dreconstructed.T


# ####################################################################################
# ### Ploting
# ####################################################################################

def set_axes_equal(ax):
	'''Make axes of 3D plot have equal scale so that spheres appear as spheres,
	cubes as cubes, etc..  This is one possible solution to Matplotlib's
	ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

	Input
	  ax: a matplotlib axis, e.g., as output from plt.gca().
	'''

	x_limits = ax.get_xlim3d()
	y_limits = ax.get_ylim3d()
	z_limits = ax.get_zlim3d()

	x_range = abs(x_limits[1] - x_limits[0])
	x_middle = np.mean(x_limits)
	y_range = abs(y_limits[1] - y_limits[0])
	y_middle = np.mean(y_limits)
	z_range = abs(z_limits[1] - z_limits[0])
	z_middle = np.mean(z_limits)

	# The plot bounding box is a sphere in the sense of the infinity
	# norm, hence I call half the max range the plot radius.
	plot_radius = 0.5*max([x_range, y_range, z_range])

	ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
	ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
	ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot3D(points3D, color='b', showOrigin=True):
    '''
        Simple plot of 3D points. (the axes are set to equal)
        - points3D numpy array (Nx3)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points3D[:,0], points3D[:,1], points3D[:,2], c=color, marker='o')

    # Mark the origin
    if showOrigin:
        ax.scatter(0, 0, 0, c='r', marker='o')

    set_axes_equal(ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()





