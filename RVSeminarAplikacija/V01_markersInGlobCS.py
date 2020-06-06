import numpy as np
import cv2
import cv2.aruco as aruco
import time

from additionalFunctions import *

# ***
# MAIN GOAL: get camera pose in marker's CS
#
# Load camera parameters (prior calibration needed) and use them to estimade pose of found ArUco markers
# This is happening live on the computer's camera (with correct camera name and camera source it would work for any calibrated
# camera)
# 
# press q to quit
# 
# press p to print marker positions in global CS (global CS in the cs of the marker with id 0)
# 
# ***




#
# Code -----------------------------------------------------------
#
recordVideo = False
videoName = 'markers02.avi'

# Settings:
numOfUsedMarkers = 4

# Load camera parameters:
# cameraName = 'EriksPhoneCam'
cameraName = 'EriksLaptopCam'
folderName = 'CameraCalibration/'
camParamsName = folderName + cameraName + '.npz'
camParams = np.load(camParamsName) 

retVal = camParams['retVal']
cameraMatrix = camParams['cameraMatrix']
distCoeffs = camParams['distCoeffs'] 
rvecsCalib = camParams['rvecs']
tvecsCalib = camParams['tvecs']
# print(retVal, cameraMatrix, distCoeffs, rvecsCalib, tvecsCalib)
print('Camera parameters loaded.')

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('trackingAndMarkers_2.mp4')



if recordVideo:
	# ###############################################################################################################
	# Default resolutions of the frame are obtained.The default resolutions are system dependent.
	# We convert the resolutions from float to integer.
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	frame_rate = int(cap.get(5))
	print(f'Started recording with frame rate {frame_rate}.')

	# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
	out = cv2.VideoWriter(videoName, cv2.VideoWriter_fourcc('M','J','P','G'), frame_rate, (frame_width,frame_height))




while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()
	#print(frame.shape) #480x640
	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
	parameters =  aruco.DetectorParameters_create()



	corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
	corners = np.array(corners)
	
	# Draw markers
	frame = aruco.drawDetectedMarkers(frame, corners, ids, (0,255,0))

	# Estimate poses
	rvecs, tvecs, objPoints= aruco.estimatePoseSingleMarkers(corners, 70, cameraMatrix, distCoeffs)

	# Draw the coordinate frames to the image
	for i in range(len(corners)):
		rvec = rvecs[i]
		tvec = tvecs[i] 
		frame = aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvec, tvec, 35)


	# ### Printing the data: (for user and debug)
	# print(corners)
	print(f'\rArUco markers found: {len(corners)}', end=',')


	# ### Marker by marker processing
	if len(corners):
		wantedMarkerId = 0 # Id that we want to take a look at
		wantedIdxs = np.nonzero(ids == wantedMarkerId)
		wantedIdxs = wantedIdxs[0] # Because array ids is actually 1D
	  
		if wantedIdxs.shape[0] > 0:
			if  wantedIdxs.shape[0] > 1:
				print('Warning! More than one marker with id {wantedMarkerId}.')
				print('The program will use one of them')
			wantedIdx = wantedIdxs[0]
		
			j = wantedIdx
			print(f' id={ids[j,0]}, tvec =', repr(tvecs[j]).replace('\n', ''), '| rvec =', repr(rvecs[j]).replace('\n', ''), '.................    ', end=' ')
		else: 
			print(f' id=/, tvec =', '[/, /, /]', '.................    ', end=' ')


	#print(rejectedImgPoints)
	# Display the resulting frame
	cv2.imshow('frame', frame)
	if recordVideo:
		out.write(frame)

	keyPressed = cv2.waitKey(1)
	if keyPressed & 0xFF == ord('p'):
		print('\nPrinting the data for this frame.')
		# Here we will calculate the positions of markers in Global CS (GLobal cs is the cs of the marker 0)
		# if marker 0 is not found we do nothing.

		# Is marker 0 seen:
		wantedMarkerId = 0 # Id that we want to take a look at
		wantedIdxs = np.nonzero(ids == wantedMarkerId)
		wantedIdxs = wantedIdxs[0] # Because array ids is actually 1D
	  
		if wantedIdxs.shape[0] == 1:
			# --- Compute marker positions in Global CS: --------------------------
			print(f'Total number of markers in the frame: {len(corners)}')
			
			# Camera in Global CS:
			wantedIdx = wantedIdxs[0]
			tvec = tvecs[wantedIdx]
			rvec = rvecs[wantedIdx]
			H_0 = vec2homo(tvec, rvec)
			camInGlob = homoInv(H_0)

			# All other markers in Global CS:
			
			foundIds = []
			foundHs = []
			markersInGlob = []
			# Compute homogenous matrices for all visible markers (H gives marker pose in relation to camera)
			for id in range(1, numOfUsedMarkers):
				markerFound, H = id2homo(id, ids, tvecs, rvecs)
				if markerFound:
					foundIds.append(id)
					foundHs.append(H)
					markersInGlob.append(np.dot(camInGlob, H))


			# Print the results:
			print('H_0: \n', H_0)
			# [print(f'H_{jj+1}:\n', foundHs[jj]) for jj in range(len(foundHs))]
			[print(f'Marker_{foundIds[jj]} in glob:\n', markersInGlob[jj]) for jj in range(len(foundHs))]	
			print(f'Ids found: {foundIds}')

			




			# *** computation and printing finished. wait a little ****************
			time.sleep(5) # Wait x seconds
		else:
			print(f'There is no marker with id {wantedMarkerId} in the frame.')

	elif keyPressed & 0xFF == ord('q'):
		break
	

# When everything done, release the capture
cap.release()
if recordVideo:
	out.release()
cv2.destroyAllWindows()