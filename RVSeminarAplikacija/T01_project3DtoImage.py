import numpy as np 
import cv2 
import matplotlib.pyplot as plt


# test script to try recoverPose function (and Ematrix) on a set of generated and projected points
# it turned into playing with projecting 3D points to a frame and making a 'screensaver' like videos.





def Rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    Rz = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
    return Rz

def Rx(theta):
    c, s = np.cos(theta), np.sin(theta)
    Rx = np.array(( (1, 0, 0), (0, c, -s), (0, s, c)  ))
    return Rx




x = np.linspace(0, 10, 6)
y = np.linspace(0, 10, 6)

xx, yy = np.meshgrid(x, y)
meshPoints2D = np.array([xx.flatten(), yy.flatten()]).transpose()
N = meshPoints2D.shape[0]
print(N)

points3D = np.zeros((3*N,3))

# Generate points resembling three cube faces (touching the same vertex)
i = 0
a = np.zeros((N, 3))
a[:,0] = meshPoints2D[:,0]
a[:,1] = meshPoints2D[:,1]
points3D[i*N:(i+1)*N] = a
i += 1

a = np.zeros((N, 3))
a[:,1] = meshPoints2D[:,0]
a[:,2] = meshPoints2D[:,1]
points3D[i*N:(i+1)*N] = a
i += 1

a = np.zeros((N, 3))
a[:,2] = meshPoints2D[:,0]
a[:,0] = meshPoints2D[:,1]
points3D[i*N:(i+1)*N] = a

# print(points3D)
referencePoints = np.copy(points3D)



# Camera data
width = 1280
height = 720
rvec = np.array([0., 0, 0])
tvec = np.array([0., 0, 0])

cameraMatrix = np.array([[1.21327941e+03, 0.00000000e+00, 6.30380082e+02],
       [0.00000000e+00, 1.21218035e+03, 3.77022123e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
distCoeffs = np.array([[ 0.11340019, -0.49745501, -0.0025145 ,  0.0007062 ,  0.3934365 ]])



# ###############################################################################################################
# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = width
frame_height = height
frame_rate = 25
videoName = 'ScreenSaverCube2.avi'
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter(videoName, cv2.VideoWriter_fourcc('M','J','P','G'), frame_rate, (frame_width,frame_height))




for t_i in range(600):
    t = t_i/25
    w = 2*np.pi*0.1


    points3D = np.copy(referencePoints)

    # Rotate points a little
    # R = Rx(0.4*w*t) #np.pi/4+np.pi)
    R = np.dot(Rz(np.sin(3*0.5*np.pi*t_i/600)), Rx(8*np.sin(0.5*np.pi*t_i/600)))
    points3D = np.dot(points3D, np.transpose(R))
    # points3D += [-5, -5, 50]
    
    

    points3D += [0, 0, 50*(1.8 - np.sin(0.5*np.pi*t_i/600))]
    # points3D += [10*np.sin(w*t), 5*np.cos(w*t), 20*(1.5 + np.sin(0.3*w*t))]
    # points3D = referencePoints + [0, 0, 50*(1.1 + np.sin(w*t))]

    # print(points3D)


    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(points3D[:,0], points3D[:,1], points3D[:,2], c='b', marker='o')
    # plt.show()


    imagePoints, _= cv2.projectPoints(points3D, rvec, tvec, cameraMatrix, distCoeffs)


    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # print(imagePoints)
    for point in np.int0(imagePoints):
                    x,y = point.ravel()
                    cv2.circle(frame,(x,y),3,(0, 255, 255),-1)



    cv2.imshow('Frame1', frame)
    out.write(frame)
   
    pressedKey =  cv2.waitKey(int(1000/50))
    if pressedKey & 0xFF ==  ord('q'):
        break
    
out.release()












