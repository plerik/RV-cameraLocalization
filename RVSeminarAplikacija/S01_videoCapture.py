import cv2
import numpy as np

# Shows the computer camera. WHen started it saves the next number of frames to file.
#
# q - quit
# r - start recording (record NframesToCapture and quit after that)



# Settings
videoName = 'tracking02.avi'
NframesToCapture = 30
recording = False




# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if (cap.isOpened() == False): 
	print("Unable to read camera feed")


# ###############################################################################################################
# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_rate = int(cap.get(5))
print(f'Opened camera with frame rate {frame_rate}.')

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter(videoName, cv2.VideoWriter_fourcc('M','J','P','G'), frame_rate, (frame_width,frame_height))




while(True):
	ret, frame = cap.read()

	if ret == True: 
		
		if recording:
				# Write the frame into the file
				out.write(frame)
				frameNumber += 1
				if frameNumber >= NframesToCapture:
						break
				

		# Display the resulting frame    
		cv2.imshow('frame',frame)

		# Press Q on keyboard to stop recording
		pressedKey = cv2.waitKey(1)
		
		if pressedKey & 0xFF == ord('r'):
				recording = True
				frameNumber = 0
		elif pressedKey & 0xFF == ord('q'):
				break

	# Break the loop
	else:
		break  

# When everything done, release the video capture and video write objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows() 

print(f'{NframesToCapture} frames recorded and saved to file {videoName}')