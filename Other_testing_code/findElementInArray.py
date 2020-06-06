import numpy as np 


a = np.array([[100 for i in range(20)]])
a = a.reshape((20,1))

print(a)

index = np.nonzero(a == 100)
print(index)
print(index[0].shape)


