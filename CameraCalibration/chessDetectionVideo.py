import numpy as np
import cv2

# ***
# Opens the video and detect chess on frames that are not blurred. Saves the detected cheesboard photos
# and optionally records the output with blur info and detected chess drawn.
#
# It's possible to set how many frames to skip for chess detection.
# 
# ***



# Chess caliber dimensions 
calibNumW = 8
calibNumH = 6
calibSquareSize = 25

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((calibNumW*calibNumH,3), np.float32)
objp[:,:2] = calibSquareSize*np.mgrid[0:calibNumW,0:calibNumH].T.reshape(-1,2)


# Open video:
videoName = 'chessVideo01.mp4'
cameraName = 'EriksPhoneCam'
blurThreshold = 90 # Blur threshold
framesToSkip = 0

recordOutput = True
outVideoName = videoName[:-4] + '_detection.avi'


folderName = 'CameraCalibration/'
folderPath = folderName + cameraName + '/'




def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()





cap = cv2.VideoCapture(folderPath + videoName)
# Check if video opened successfully
if (cap.isOpened() == False): 
    print("Unable to open video")


if recordOutput:
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_rate = int(cap.get(5))
    print(f'Recording video with frame rate {frame_rate}.')

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter(folderPath + outVideoName, cv2.VideoWriter_fourcc('M','J','P','G'), frame_rate, (frame_width,frame_height))






photoNum = 0

while(True):
    # Capture frame-by-frame
    for i in range(framesToSkip):
        ret, frame = cap.read()
    ret, frame = cap.read()
    
    if not ret:
        # video ended
        break

    # Our operations on the frame come here
    img = np.copy(frame)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


    # Check if image is blurry
    fm = variance_of_laplacian(gray)
    
    ret = False
    if fm > blurThreshold:       
        # if the focus measure is less than the supplied threshold,
        # then the image should be considered "blurry" 


        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (calibNumW,calibNumH), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (calibNumW,calibNumH), corners2,ret)

        # show the image
        text = "Not Blurry"
        cv2.putText(img, "{}: {:.2f}".format(text, fm), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
    
    else:
        text = '    Blurry'
        cv2.putText(img, "{}: {:.2f}".format(text, fm), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)


    cv2.imshow('img',img)
    if recordOutput:
        out.write(img)

    if ret == True:
        # Save the photo'
        cv2.imwrite(folderPath + f"chess{photoNum:03}.jpg", frame)  # save the original frame (not to have chessboard drawn on top)
        print(f'Photo saved. That was photo number {photoNum}.')
        photoNum += 1



    pressedKey = cv2.waitKey(1) # (int(1000/30))
    if pressedKey & 0xFF == ord('q'):
        break

# When everything done, release the capture

cap.release()
print(f'Chess extraction finished. Saved {photoNum-1} photos.')
if recordOutput:
    out.release()
    print(f'Video recorded to {folderPath + outVideoName}')
cv2.destroyAllWindows()