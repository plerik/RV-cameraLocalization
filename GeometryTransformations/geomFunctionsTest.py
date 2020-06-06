from geomFunctions import *


np.set_printoptions(precision=2)

# Test data:
tvec1 = np.array([[0, 0, 100]])  # translation vector
rvec1 = np.array([[np.pi, 0, 0]]) # rotation vector

tvec2 = np.array([[0, 50, 80]])  # translation vector
rvec2 = np.array([[np.pi, 0, 0]]) # rotation vector



H1 = vec2homo(tvec1, rvec1)
invH1 = homoInv(H1)
print('invH (also cam in GLOB):\n', invH1)


H2 = vec2homo(tvec2, rvec2)
H2_glob = np.dot(invH1, H2)
print('H2_glob:')
print(H2_glob)




