import cv2
import numpy as np

# Shows the computer camera. WHen started it saves the next number of frames to file.
#
# q - quit
#
# [when playbackMode = 'frame']
# i - info about the frame
# (any key (including i) beside q) - next frame



# Settings
videoName = 'tracking01.avi'
playbackModes = ['play', 'frame']
playbackMode = playbackModes[1]



# Create a VideoCapture object
cap = cv2.VideoCapture(videoName)

# Check if camera opened successfully
if (cap.isOpened() == False): 
  print("Unable to open video")

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_rate = int(cap.get(5))
print(f'Opened video with frame rate {frame_rate}.')



while(True):
  ret, frame = cap.read()

  if ret == True: 
    

    # Display the resulting frame    
    cv2.imshow('frame',frame)

    # Press Q on keyboard to quit
    if playbackMode == 'play':
        pressedKey = cv2.waitKey(int(1000/frame_rate))
        if pressedKey & 0xFF == ord('q'):
            break
    elif playbackMode == 'frame':
        pressedKey = cv2.waitKey(0)
        if pressedKey & 0xFF == ord('i'):
            print(f'Frame number = {int(cap.get(1))}')        
        elif pressedKey & 0xFF == ord('q'):
            break

  # Break the loop
  else:
    break  

# When everything done, release the video capture and video write objects
cap.release()


# Closes all the frames
cv2.destroyAllWindows() 

