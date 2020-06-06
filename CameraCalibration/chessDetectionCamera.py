import numpy as np
import cv2

# ***
# Runs the computers cam and show the found chessboard on the video.
# when you press w (and if chessboard is found) the photo is saved (raw without the chessboard drawn over)
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




cap = cv2.VideoCapture(0)
photoNum = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    img = np.copy(frame)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (calibNumW,calibNumH), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (calibNumW,calibNumH), corners2,ret)

    cv2.imshow('img',img)



    pressedKey = cv2.waitKey(1)
    
    if pressedKey & 0xFF == ord('w'):
        # w -> take and save a photo (if chessboard was found)
        if ret == True:
            # Save the photo'
            cv2.imwrite(f"chess{photoNum:03}.jpg", frame)  # save the original frame (not to have chessboard drawn on top)
            print(f'Photo saved. That was photo number {photoNum}.')
            photoNum += 1

    elif pressedKey & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()