import numpy as np
import cv2
import cv2.aruco as aruco
 
 
'''
    drawMarker(...)
        drawMarker(dictionary, id, sidePixels[, img[, borderBits]]) -> img
'''
 
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
print(aruco_dict)

codeNum = 4 # Number of genereted codes

for i in range(codeNum):
    # second parameter is id number
    # last parameter is total image size
    img = aruco.drawMarker(aruco_dict, i, 700)
    cv2.imwrite(f"6x6marker_{i:03d}.jpg", img)
    
    cv2.imshow(f'frame id = {i}',img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()