import numpy as np
import cv2
import cv2.aruco as aruco



path = 'singlemarkersoriginal.jpg'

# Load an image
frame = cv2.imread(path)

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
print(corners)

#It's working.
# my problem was that the cellphone put black all around it. The alrogithm
# depends very much upon finding rectangular black blobs

frame = aruco.drawDetectedMarkers(frame, corners, ids, (0,255,0))

#print(rejectedImgPoints)
# Display the resulting frame
cv2.imshow('frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

