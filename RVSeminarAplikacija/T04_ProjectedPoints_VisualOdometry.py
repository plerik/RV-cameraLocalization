import numpy as np 
import cv2 
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from additionalFunctions import *


# Test script to try recoverPose function (and Ematrix) on a set of generated and projected points
# - this version uses one camera which is moving in space - we are estimatind the movement

# We move the camera around and compare the actual movement to estimated movement from visual 
# odometry.
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

# Move points (in front of camera)
R = rotMat(np.pi/4, 1)
points3D = np.dot(points3D, np.transpose(R))
points3D += [-5, -5, 40]



# #######################
# ---   Camera data   ---
width = 1280
height = 720
cameraMatrix = np.array([[1.21327941e+03, 0.00000000e+00, 6.30380082e+02],
	   [0.00000000e+00, 1.21218035e+03, 3.77022123e+02],
	   [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
distCoeffs = np.array([[ 0.11340019, -0.49745501, -0.0025145 ,  0.0007062 ,  0.3934365 ]])



# Camera 1 and 2 represent different times of the taken picture
# camera 2 is the current frame, camera 1 is the frame before.



for k in range(0, 10+1):

	# Camera 1 position 
	# rvec1 = np.array([[0., 0, 0]])
	# tvec1 = np.array([[0., 0, 0]])
	# rvec1 *= -1
	# tvec1 = -tvec1.T

	# Camera 2 position
	rvec2 = np.array([[0., -0.3*np.cos(2*np.pi*k/10), 0]])
	tvec2 = np.array([[20*np.cos(2*np.pi*k/10), 0, 10*np.sin(2*np.pi*k/10)]]).T
	rvec2 *= -1
	tvec2 = -tvec2

	R2, _ = cv2.Rodrigues(rvec2)


	# #############################
	# --- Generate image points --- (project 3D points to images) 

	
	# Project points to camera 2
	imagePoints2, _= cv2.projectPoints(points3D, rvec2, tvec2, cameraMatrix, distCoeffs)
	imagePoints2 = np.squeeze(imagePoints2)
	frame2 = np.zeros((height, width, 3), dtype=np.uint8)
	for point in np.int0(imagePoints2):
					x,y = point.ravel()
					cv2.circle(frame2,(x,y),3,(150, 200, 0),-1)
	cv2.imshow('Frame2', frame2)


	if k < 1:
		# Estimated:
		He_list = []
		He_list.append(np.eye(4))

		HeAbs_list = []
		# reference absolute pose
		Habs = np.hstack((R2,tvec2))
		Habs = np.vstack((Habs, [0, 0, 0, 1]))
		# start initilanization of the pose
		HeAbs_list.append(Habs)

		# Reference:
		Habs_list = []
		Habs_list.append(Habs)
	else:
		# ####################################
		# ### recoverPose and E estimation ---
		# #
		#

		# --- Visual odometry:
		p1 = imagePoints1
		p2 = imagePoints2

	



		E, mask = cv2.findEssentialMat(p1, p2, cameraMatrix) 
		retval, Re, te, mask = cv2.recoverPose(E, p1, p2, cameraMatrix)
		print('----', mask)

		# --- 3D points reconstruction
		pointsEstimated = points3Dreconstruction(p1, p2, Re, te, cameraMatrix)
		print('p1.shape:', p1.shape)
		print('estimated.shape: ', pointsEstimated.shape)


		# -- scale
		distance3D = np.linalg.norm(points3D[1] - points3D[0])
		distanceEstimated = np.linalg.norm(pointsEstimated[1,:3] - pointsEstimated[0,:3])

		scaleFactor = distance3D/distanceEstimated
		pointsEstimated *= scaleFactor
		te *= scaleFactor


		# ------------------------
		# --- Compute the results:
		
		# We want to have pose expressed as how much camera2 is moved te (in global 
		# CS [defined as CS of camera 1]) from origin and rotated Re
		He = np.hstack((Re,te))
		He = np.vstack((He, [0, 0, 0, 1]))

		# Estimation for absolute pose
		# ---------------# HeAbs = np.dot(HeAbs_list[-1], He)
		HeAbs = np.dot(Habs_list[0], He)


		# reference absolute pose
		Habs = np.hstack((R2,tvec2))
		Habs = np.vstack((Habs, [0, 0, 0, 1]))

		# H relative from pose of the old photo (according to reference)
		print('----------------')
		print('k = ', k)
		# print('HAbs list \n', Habs_list)
		# print('HeAbs list \n',HeAbs_list)
		
		Href = np.dot(homoInv(Habs_list[-1]), Habs)

		# ----------------------
		# --- Print the results:
		
		print('HeAbs (estimated):\n', HeAbs)
		print('Habs (reference):\n', Habs)

		# print('He (estimated):\n', He)
		# print('Href (reference):\n', Href)
		# print('Href-He (error):\n', Href-He)

		# Store matrices in lists
		He_list.append(He)
		HeAbs_list.append(HeAbs)
		Habs_list.append(Habs)

		




		#
		# #
		# ### recoverPose and E estimation ---
		# ####################################

	if k == 0:
		imagePoints1 = imagePoints2

	pressedKey = cv2.waitKey(10)
	if pressedKey & 0xFF == ord('q'):
		break


	# --- Ploting points and camera CSs ---

	# Show the actual 3D setup
	if True: 
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')

		ax.scatter(points3D[:,0], points3D[:,1], points3D[:,2], c='b', marker='o')
		

		# Plot reference points
		for jj in range(k+1):
			H = Habs_list[jj]
			tvecE = -H[0:3, 3]
			ax.scatter(tvecE[0], tvecE[1], tvecE[2], c='g')
		# Plot reference coordinate frame
		plotCoordinateFrame(ax, R2.T, -tvec2, 'g')





		# Plot estimated points
		for jj in range(k+1):
			H = HeAbs_list[jj]
			tvecE = -H[0:3, 3]
			ax.scatter(tvecE[0], tvecE[1], tvecE[2], c='k')
		# Plot estimated coordinate frame
		plotCoordinateFrame(ax, H[:3,:3].T, tvecE, 'y')


		# Mark the origin
		ax.scatter(0, 0, 0, c='r', marker='o')

		


		set_axes_equal(ax)
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		ax.set_zlabel("z")
		plt.show()

	# Show the estimated 3D points
	if k > 1 and True:
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')

		ax.scatter(pointsEstimated[:,0], pointsEstimated[:,1], pointsEstimated[:,2], c='y', marker='o')
		ax.scatter(0, 0, 0, c='r', marker='.')


		# plotCoordinateFrame(ax, R1.T, -tvec1, 'r')
		plotCoordinateFrame(ax, Re.T, -te, 'g')

		set_axes_equal(ax)
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		ax.set_zlabel("z")
		plt.show()







