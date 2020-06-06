import cv2
import numpy as np

# Shows the computer camera. WHen started it saves the next number of frames to file.
#
# q - quit
#
# [when playbackMode = 'frame']
# i - info about the frame
# (any key (including i) beside q) - next frame
#
# r - reset (find new features to track)



# ### Settings
inVideoName = 'tracking01.avi'
outVideoName = 'LK_demo01.avi'
playbackModes = ['play', 'frame']
playbackMode = playbackModes[0]
recording = False



# ### Parameters
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
					   qualityLevel = 0.03,
					   minDistance = 7,
					   blockSize = 7 )
# My old way: prevPts = cv2.goodFeaturesToTrack(gray,5,0.01,10)

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
				  maxLevel = 3,
				  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# My old way: currPts, status, err = cv2.calcOpticalFlowPyrLK(gray, grayOld, prevPts, None) 








######### - Opening the video:
##
#
# Create a VideoCapture object
# cap = cv2.VideoCapture(videoName)
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if (cap.isOpened() == False): 
	print("Unable to open video")

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
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


while(True):

	ret, frame = cap.read()
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	if ret == True: 
		######### - Processing of the video
		##
		#


		if findFeaturesToTrack:
			### find features to track
			prevPts = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
			
			for point in np.int0(prevPts):
				x,y = point.ravel()
				cv2.circle(frame,(x,y),3,(0, 255, 255),-1)
			findFeaturesToTrack = False

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

			# # Draw old points
			# for point in np.int0(prevPts):
			#     x,y = point.ravel()
			#     cv2.circle(frame,(x,y),3,(150, 150, 250),-1)

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
		if recording:
			out.write(frame)
		frame = cv2.flip(frame, 1)   
		cv2.imshow('frame',frame)
		
		grayOld = gray 

		# Press Q on keyboard to quit
		if playbackMode == 'play':
			pressedKey = cv2.waitKey(int(1000/frame_rate))
		elif playbackMode == 'frame':
			pressedKey = cv2.waitKey(0)


		if pressedKey & 0xFF == ord('i'):
			print(f'Frame number = {int(cap.get(1))}')
		elif pressedKey & 0xFF == ord('r'):
			findFeaturesToTrack = True      
		elif pressedKey & 0xFF == ord('q'):
			break

		
		#
		##
		###################################

	# Break the loop
	else:
		break  

# When everything done, release the video capture and video write objects
cap.release()
if recording:
	out.release()
# Closes all the frames
cv2.destroyAllWindows() 

