import numpy as np
import cv2
import cv2.aruco as aruco


# ***
# Load camera parameters (prior calibration needed) and use them to estimade pose of found ArUco markers
# This is happening live on the computer's camera (with correct camera name and camera source it would work for any calibrated
# camera)
# 
# pres q to quit
# 
# ***



# Load camera parameters:
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


cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #print(frame.shape) #480x640
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters =  aruco.DetectorParameters_create()

    #print(parameters)

    '''    detectMarkers(...)
        detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedI
        mgPoints]]]]) -> corners, ids, rejectedImgPoints
        '''
        #lists of ids and the corners beloning to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    # print(corners)
    print(f'ArUco markers found: {len(corners)}')

    #It's working.
    # my problem was that the cellphone put black all around it. The alrogithm
    # depends very much upon finding rectangular black blobs

    # gray = aruco.drawDetectedMarkers(gray, corners, ids, (0,255,0))
    frame = aruco.drawDetectedMarkers(frame, corners, ids, (0,255,0))


    
    rvecs, tvecs, objPoints= aruco.estimatePoseSingleMarkers(corners, 70, cameraMatrix, distCoeffs)

    for i in range(len(corners)):
        rvec = rvecs[i]
        tvec = tvecs[i] 
        frame = aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvec, tvec, 35)
    


    #print(rejectedImgPoints)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()