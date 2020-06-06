import numpy as np 
import cv2 
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


# Test script to try recoverPose function (and Ematrix) on a set of generated and projected points
# Two Camera positions and 3D points are specified (in 3D). script projects the points onto
# camera images and from points on that images it computes the estimation for relative pose
# between the cameras and estimation for 3D position of the points.
# (visualization: 3D view of the initial setup, both camera frames, 3D view of the estimation)
# 
# For the scale estimation we use the first two points(in initial cloud and after estimation)
# and make the distance between them the same.


def rotMat(theta, axis):
	c, s = np.cos(theta), np.sin(theta)
	if axis == 0:
		R = np.array(( (1, 0, 0), (0, c, -s), (0, s, c) ))
	elif axis == 1:
		R = np.array(( (c, 0, s), (0, 1, 0), (-s, 0, c) ))
	elif axis == 2:
		R = np.array(( (c, -s, 0), (s, c, 0), (0, 0, 1) ))
	else:
		print(f'{axis} is invalid parameter for axis. Valid are 0, 1 or 2.')
		return 0
	return R


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

def plotCoordinateFrame(ax, rotation, origin, color):
    for i in range(3):
        lines = [[float(origin[j]), float(origin[j] + 20*rotation[j,i])] for j in range(3)]
        a = Arrow3D(lines[0], lines[1], lines[2], mutation_scale=20,
                    lw=2, arrowstyle="-|>", color=color)
        ax.add_artist(a)


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


np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


# ##########################
# --- Generate 3D points ---
# Generate points resembling three cube faces (touching the same vertex)
x = np.linspace(0, 10, 6)
y = np.linspace(0, 10, 6)

xx, yy = np.meshgrid(x, y)
meshPoints2D = np.array([xx.flatten(), yy.flatten()]).transpose()
N = meshPoints2D.shape[0]

points3D = np.zeros((3*N,3))

i = 0
a = np.zeros((N, 3))
a[:,0] = meshPoints2D[:,0]
a[:,1] = meshPoints2D[:,1]
points3D[i*N:(i+1)*N] = a
i += 1

a = np.zeros((N, 3))
a[:,1] = meshPoints2D[:,0]
a[:,2] = meshPoints2D[:,1]
points3D[i*N:(i+1)*N] = a
i += 1

a = np.zeros((N, 3))
a[:,2] = meshPoints2D[:,0]
a[:,0] = meshPoints2D[:,1]
points3D[i*N:(i+1)*N] = a
# print(points3D)


# #######################
# ---   Camera data   ---
width = 1280
height = 720
cameraMatrix = np.array([[1.21327941e+03, 0.00000000e+00, 6.30380082e+02],
	   [0.00000000e+00, 1.21218035e+03, 3.77022123e+02],
	   [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
distCoeffs = np.array([[ 0.11340019, -0.49745501, -0.0025145 ,  0.0007062 ,  0.3934365 ]])

# Camera 1 position
rvec1 = np.array([[0., 0, 0]])
tvec1 = np.array([[0., 0, 0]])
rvec1 *= -1
tvec1 = -tvec1.T
# Camera 2 position
alpha = 0 #np.pi/3
rvec2 = np.array([[0., alpha, 0]])
# rvec2 = np.array([[0., 0, 0]])
tvec2 = np.array([[10., 0, 0]]).T
# tvec2 = np.array([[0., 0, -40]]).T
# tvec2 = np.dot(rotMat(alpha, 1), tvec2) + np.array([[0., 0, 40]]).T
print(tvec2)
rvec2 *= -1
tvec2 = -tvec2

R1, _ = cv2.Rodrigues(rvec1)
R2, _ = cv2.Rodrigues(rvec2)


# #############################
# --- Generate image points --- (project 3D points to images) 
# Move points (in front of camera)
R = rotMat(np.pi/4, 1)
points3D = np.dot(points3D, np.transpose(R))
points3D += [-5, -5, 40]

# Project points to camera 1
imagePoints1, _= cv2.projectPoints(points3D, rvec1, tvec1, cameraMatrix, distCoeffs)
imagePoints1 = np.squeeze(imagePoints1)
frame1 = np.zeros((height, width, 3), dtype=np.uint8)
for point in np.int0(imagePoints1):
				x,y = point.ravel()
				cv2.circle(frame1,(x,y),3,(0, 255, 255),-1)
cv2.imshow('Frame1', frame1)
# Project points to camera 2
print('rvec2 = ', rvec2)
print('tvec2 = ', tvec2)
imagePoints2, _= cv2.projectPoints(points3D, rvec2, tvec2, cameraMatrix, distCoeffs)
imagePoints2 = np.squeeze(imagePoints2)
frame2 = np.zeros((height, width, 3), dtype=np.uint8)
for point in np.int0(imagePoints2):
				x,y = point.ravel()
				cv2.circle(frame2,(x,y),3,(0, 255, 255),-1)
cv2.imshow('Frame2', frame2)


# ####################################
# ### recoverPose and E estimation ---
# #
#


# --- Visual odometry:
p1 = imagePoints1
p2 = imagePoints2
E, mask = cv2.findEssentialMat(p1, p2, cameraMatrix) 
retval, Re, te, mask = cv2.recoverPose(E, p1, p2, cameraMatrix)

# We want to have pose expressed as how much camera2 is moved te (in global 
# CS [defined as CS of camera 1]) from origin and rotated Re
He = np.hstack((Re,te))
He = np.vstack((He, [0, 0, 0, 1]))


# --- 3D points reconstruction


M1 = np.dot(cameraMatrix, np.hstack((R1, tvec1)))
print('M1 = \n', M1)
M2 = np.dot(cameraMatrix, np.hstack((Re, te)))
print('M2 = \n', M2)




print('type p2:', p2.dtype)
print('shape p2.T:', p2.transpose().shape)
pointsEstimated = cv2.triangulatePoints(M1, M2, p1.T, p2.T)
# vec1 = np.array([height/2, width/2], dtype=np.float)
# pointsEstimated = cv2.triangulatePoints(M1, M2, vec1, vec1)
pointsEstimated /= pointsEstimated[3]
pointsEstimated = pointsEstimated.transpose()

print('pointsEstimated: \n', pointsEstimated[:30])



# dimensional accuracy (scale)
distance3D = np.linalg.norm(points3D[1] - points3D[0])
distanceEstimated = np.linalg.norm(pointsEstimated[1,:3] - pointsEstimated[0,:3])

scaleFactor = distance3D/distanceEstimated
pointsEstimated *= scaleFactor
te *= scaleFactor

# print(pointsEstimated.shape)
# print(pointsEstimated)




# Print the results:
print('He (estimated):\n', He)
print('Re (R estimated):\n', Re)
print('R2 (R of camera 2):\n', R2)

# print('te (scaled):\n', te*tvec2[0]/te[0])
print('te (estimated):\n', te)
print('tvec2 (translation of camera 2):\n', tvec2)



#
# #
# ### recoverPose and E estimation ---
# ####################################






# --- Ploting points and camera CSs ---

# Show the actual 3D setup
if True: 
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	ax.scatter(points3D[:,0], points3D[:,1], points3D[:,2], c='b', marker='o')
	ax.scatter(0, 0, 0, c='r', marker='.')


	print(R2.T)
	print('-----------\n', tvec1)
	print('-----------\n', tvec2)
	plotCoordinateFrame(ax, R1.T, -tvec1, 'r')
	plotCoordinateFrame(ax, R2.T, -tvec2, 'g')

	set_axes_equal(ax)
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_zlabel("z")
	plt.show()

# Show the estimated 3D points
if True:
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	ax.scatter(pointsEstimated[:,0], pointsEstimated[:,1], pointsEstimated[:,2], c='y', marker='o')
	ax.scatter(0, 0, 0, c='r', marker='.')


	plotCoordinateFrame(ax, R1.T, -tvec1, 'r')
	plotCoordinateFrame(ax, Re.T, -te, 'g')

	set_axes_equal(ax)
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_zlabel("z")
	plt.show()







