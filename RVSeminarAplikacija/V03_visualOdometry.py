import cv2
import numpy as np

from additionalFunctions import *

# Opens the video from file. Tracks points with Lucas-Kanade and estimates the
# camera movement using fundamental matrix.
#
# q - quit
#
# [when playbackMode = 'frame']
# i - info about the frame
# (any key (including i) beside q) - next frame



# ------------------------notes to self:
# za naprej delat
# potem pa naredi nacin ki bo iskal nove tocke za jim sledit. za zacetek
# kar v usakem frejmu good features to track (alpa usak 5. frame na novo recimo)
# alpa ko pade trenutno stevilo pod neko mejo, najdes na novo nove
# potem bi pa lahko samo dopolnjevali ta seznam

#### ************ Nadaljevanje: zdej je tracking in pose estimation za silo
# narejen. ((nima pa se upostevanja skale!!)) 
# poleg skale, je treba to se v eno pametno obliko spravit (iskanje corespondencnih
# tock, sledenje, kako pogosto estimatamo pozo, a uporabimo vecji nabor tock...)




# ### --- Settings ---
videosList = ['tracking01.mp4', 'tracking02.mp4', 'tracking03.mp4', 
				'trackingAndMarkers_1.mp4', 'trackingAndMarkers_2.mp4',
				'trackingAndMarkers_3.mp4']
videoName = videosList[4]
playbackModes = ['play', 'frame']
playbackMode = playbackModes[1]
recording = True
outVideoName = 'LK_tracking02.avi'

# ### --- Parameters ---
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 200,
					   qualityLevel = 0.01,
					   minDistance = 7,
					   blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
				  maxLevel = 3,
				  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


# ### --- Load camera parameters ---
cameraName = 'EriksPhoneCam'
folderName = 'CameraCalibration/'
camParamsName = folderName + cameraName + '.npz'
camParams = np.load(camParamsName) 
#
retVal = camParams['retVal']
cameraMatrix = camParams['cameraMatrix']
distCoeffs = camParams['distCoeffs'] 

print(repr(cameraMatrix))
print(repr(distCoeffs))
#
print('Camera parameters loaded.')


np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


######### --- Opening the video ---
##
#
# Create a VideoCapture object
cap = cv2.VideoCapture(videoName)

# Check if camera opened successfully
if (cap.isOpened() == False): 
	print("Unable to open video")

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# print(frame_width)
# print(frame_height)
frame_rate = int(cap.get(5))
print(f'Opened video with frame rate {frame_rate}.')
#
##
#############################

# ### Recording: (if recording is set to True)
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
if recording:
	out = cv2.VideoWriter(outVideoName, cv2.VideoWriter_fourcc('M','J','P','G'), frame_rate, (frame_width,frame_height))




oldFrame = 0
findFeaturesToTrack = True
homoPoses = []
homoPoses.append(np.eye(4))

k = 0
stopPlayback = False
while(True):

	ret, frame = cap.read()
	
	

	
	if ret == True: 
		######### - Processing of the video
		##
		#
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


		if findFeaturesToTrack:
			### find features to track
			
			centerAreaMask = np.zeros((frame_height, frame_width), dtype=np.uint8)
			maskRadius = 500
			# int(frame_height/2)-maskRadius:int(frame_height/2)+maskRadius
			centerAreaMask[:, \
							int(frame_width/2)-maskRadius:int(frame_width/2)+maskRadius] = 255
			prevPts = cv2.goodFeaturesToTrack(gray, mask=centerAreaMask, **feature_params)
			
			frame0_raw = np.copy(frame)

			for point in np.int0(prevPts):
				x,y = point.ravel()
				cv2.circle(frame,(x,y),3,(0, 255, 255),-1)
			findFeaturesToTrack = False

			p0 = prevPts
			

		else:
			### track features
			
			# Calculate optical flow (i.e. track feature points)
			currPts, status, err = cv2.calcOpticalFlowPyrLK(grayOld, gray, prevPts, None, **lk_params) 

			# Sanity check
			assert prevPts.shape == currPts.shape 

			# Filter only valid points
			idx = np.where(status==1)[0]
			
			prevPts = prevPts[idx]
			currPts = currPts[idx]
			p0 = p0[idx]


			if k == 250 and False:
				# ### Visual odometry:
				p1 = np.copy(p0)
				p2 = currPts


				# Save frames and points to file:
				path = 'RVSeminarAplikacija/twoCameraViews11.npz'
				np.savez(path, frame1=frame0_raw, frame2=frame, p1=p1, 
									p2=p2, cameraMatrix=cameraMatrix)
				print(f'2 Frames and corresponding points saved to: {path} ')



				# Draw new points
				frame0 = np.copy(frame0_raw)
				for point in np.int0(p1):
					x,y = point.ravel()
					cv2.circle(frame0,(x,y),3,(150, 150, 250),-1)

				for point in np.int0(p2):
					x,y = point.ravel()
					cv2.circle(frame,(x,y),3,(150, 150, 250),-1)



				cv2.imshow('First Frame', frame0)
				cv2.imshow('Current Frame', frame)

				
				

				cv2.waitKey(0)
				break



				E, maskEs = cv2.findEssentialMat(p1, p2, cameraMatrix) 
				retval, R, t, maskRp = cv2.recoverPose(E, p1, p2, cameraMatrix)

				# Remove 'bad' points:
				print(f'shape p1 before: {p1.shape}')
				idx = np.where(maskEs==1)[0]
				p1 = p1[idx]
				p2 = p2[idx]
				print(f'shape p1 after: {p1.shape}')

				


				# --- 3D points reconstruction
				pointsEstimated = points3Dreconstruction(p1, p2, R, t, cameraMatrix)

				
				print('maskEs', maskEs)
				print('maskRp', maskRp)
				

				print(pointsEstimated)
				print('shape:', pointsEstimated.shape)
				plot3D(pointsEstimated)








				H = np.hstack((R,t))
				H = np.vstack((H, [0, 0, 0, 1]))
				
				totalH = np.dot(homoPoses[-1], H)
				homoPoses.append(totalH)
				# print('totalH:\n', totalH)
				print('H:\n', H)

				# # Draw old points
				# for point in np.int0(prevPts):
				#     x,y = point.ravel()
				#     cv2.circle(frame,(x,y),3,(150, 150, 250),-1)
				stopPlayback = True
			else:
				stopPlayback = False






			# Draw new points
			for point in np.int0(currPts):
				x,y = point.ravel()
				cv2.circle(frame,(x,y),3,(150, 250, 150),-1)


			prevPts = currPts

		#
		##
		###################################



		######### - Display and control
		##
		#


		# Display the resulting frame  
		print(f'k = {k}')  
		cv2.imshow('frame',frame)
		if recording:
			out.write(frame)
		grayOld = gray 

		# Press Q on keyboard to quit
		if playbackMode == 'play':
			pressedKey = cv2.waitKey(int(1000/frame_rate))
			if pressedKey & 0xFF == ord('q'):
				break
		elif playbackMode == 'frame':

			wait_time = 0 if stopPlayback else 25
			
			pressedKey = cv2.waitKey(wait_time)
			if pressedKey & 0xFF == ord('i'):
				print(f'Frame number = {int(cap.get(1))}')        
			elif pressedKey & 0xFF == ord('q'):
				break

		#
		##
		###################################

	# Break the loop
	else:
		break
	k += 1

# When everything done, release the video capture and video write objects
cap.release()
if recording:
	out.release()
# Closes all the frames
cv2.destroyAllWindows() 

