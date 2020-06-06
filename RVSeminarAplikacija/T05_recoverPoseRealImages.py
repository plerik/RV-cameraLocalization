import cv2
import numpy as np
import cv2.aruco as aruco

from additionalFunctions import *


# --------------------- Aruco Markers --------------------------
def cornerPointsAndH(frame1, markerOfInterest=0, drawFrame=True):
    '''
        finds one arucoMarker of selected Id and returns its pose
        in camera CS.
        It also returns corners of all markers found in the frame
        (sorted with id's ascending)
    '''
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters =  aruco.DetectorParameters_create()
    # Find Markers
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    corners = np.array(corners)
    # Sort the markes based on Id
    indices = np.argsort(ids[:,0])
    ids_sorted = ids[indices]   
    corners = corners[indices]

    # Estimate poses
    rvecs, tvecs, objPoints= aruco.estimatePoseSingleMarkers(corners, 70, cameraMatrix, distCoeffs)
    # Take out one marker Data (pose as Homogenous matrix)
    
    
    marker0idx = 0
    markersWithThisId = (ids == marker0idx).sum()
    if markersWithThisId < 1:
        print(f'Warning: No markers with id {marker0idx} found!')
    elif markersWithThisId > 1:
        print(f'Warning: There are multiple ({markersWithThisId}) markers \
                                                    with id {marker0idx} !')

    H_marker0 = vec2homo(tvecs[marker0idx], rvecs[marker0idx])

    # Draw the coordinate frame to the image
    if drawFrame:
        frame1 = aruco.drawAxis(frame1, cameraMatrix, distCoeffs, 
                        rvecs[marker0idx], tvecs[marker0idx] , 35)
    
    return corners, H_marker0




# ### --- Load frames and points ---
path = 'RVSeminarAplikacija/twoCameraViews3.npz'
data = np.load(path) 
# Unpack
frame1 = data['frame1']
frame2 = data['frame2']
p1 = data['p1']
p2 = data['p2']
# cameraMatrix = data['cameraMatrix']
print('Data loaded.')



# Load camera parameters:
cameraName = 'EriksPhoneCam'
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

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)




# Camera1 position in marker CS
corners1, H_frame1 = cornerPointsAndH(frame1, markerOfInterest=0, drawFrame=True)
Cam1_inM0 = homoInv(H_frame1)
print('Cam1_inM0 (camera1 in M0 (global CS)):\n', Cam1_inM0)

# Camera2 position in marker CS
corners2, H_frame2 = cornerPointsAndH(frame2, markerOfInterest=0, drawFrame=True)
Cam2_inM0 = homoInv(H_frame2)
print('Cam2_inM0 (camera2 in M0 (global CS)):\n', Cam2_inM0)

# Camera2 position in camera1 CS
Cam2_inCam1 = np.dot(homoInv(Cam1_inM0), Cam2_inM0)
print('*** H_markers (Cam2_inCam1 - camera2 in camera1): ***\n', Cam2_inCam1)



# Show loaded points
for point in np.int0(p1):
    x,y = point.ravel()
    cv2.circle(frame1,(x,y),3,(150, 150, 250),-1)

for point in np.int0(p2):
    x,y = point.ravel()
    cv2.circle(frame2,(x,y),3,(150, 150, 250),-1)


corners1 = np.reshape(corners1, (16, 2))
corners2 = np.reshape(corners2, (16, 2))


# Show detected corners
for point in np.int0(corners1):
    x,y = point.ravel()
    cv2.circle(frame1,(x,y),3,(250, 150, 50),-1)

for point in np.int0(corners2):
    x,y = point.ravel()
    cv2.circle(frame2,(x,y),3,(250, 150, 50),-1)




# --------------------- Pose estimation --------------------------
p1 = np.squeeze(p1)
p2 = np.squeeze(p2)

E, maskEs = cv2.findEssentialMat(p1, p2, cameraMatrix) 
retval, R, t, maskRp = cv2.recoverPose(E, p1, p2, cameraMatrix)

# Remove 'bad' points:
# print(f'shape p1 before bad point removal: {p1.shape}')
idx = np.where(maskEs==1)[0]
p1 = p1[idx]
p2 = p2[idx]
# print(f'shape p1 after bad point removal: {p1.shape}')


R = R.T
t = -t
Rabs, tabs = homo2Rt(Cam2_inCam1)

He = np.hstack((R,t))
He = np.vstack((He, [0, 0, 0, 1]))
print('He:\n', He)


# print('-------------------------------------------------------')
# --- 3D points reconstruction
pointsEcorners = points3Dreconstruction(corners1, corners2, R.T, -t, cameraMatrix)
pointsEstimated = points3Dreconstruction(p1, p2, R.T, -t, cameraMatrix)
# pointsEstimated = points3Dreconstruction(corners1, corners2, Rabs.T, -tabs, cameraMatrix)
# pointsEstimated = points3Dreconstruction(p1, p2, Rabs.T, -tabs, cameraMatrix)

# print('pointsEstimated 3D shape:', pointsEstimated.shape)
# print('pointsEstimated 3D:', pointsEstimated)
# print('-------------------------------------------------------')


# Mark 'good' points (on the camera images):
for point in np.int0(p1):
    x,y = point.ravel()
    cv2.circle(frame1,(x,y),3,(50, 200, 0),-1)

for point in np.int0(p2):
    x,y = point.ravel()
    cv2.circle(frame2,(x,y),3,(50, 200, 0),-1)



# Scale:
cornerDistances = []
# print('Computing distances:')
for i in range(pointsEcorners.shape[0]):

    c1 = pointsEcorners[i,:3]
    if i % 4 == 3:
        # take the first corner of current marker
        c2 = pointsEcorners[i-3,:3]
    else:
        # take the next corner (of the same marker)
        c2 = pointsEcorners[i+1,:3]
    cornerDistances.append(np.linalg.norm(c2 - c1))

cornerDistances = np.array(cornerDistances)
scaleFactor = 70/np.mean(cornerDistances)


pointsEcorners *= scaleFactor
pointsEstimated *= scaleFactor
t *= scaleFactor

He = np.hstack((R,t))
He = np.vstack((He, [0, 0, 0, 1]))
print('He (estimated) after scale:\n', He)


print('He - H_markers:\n', He-Cam2_inCam1)









cv2.imshow('Frame 1', frame1)
cv2.imshow('Frame 2', frame2)

# cv2.waitKey(0)
plot3D(pointsEstimated, showOrigin=False)
plot3D(pointsEcorners, showOrigin=False)





