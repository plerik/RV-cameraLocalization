import numpy as np 

# We'll sort based on a
a = np.array([[10, 0, 20, 40, 30]])
a = a.reshape((5,1))

# we want to sort 
aa = np.array([[10, 0, 20, 1], [0, 0, 20,2], [20, 0, 20, 3], [40, 0, 20, 4], [30, 0, 20, 5]])



print('base the sort on:', a)
print('sort aa as well:\n', aa)

sortIds = a[:,0].argsort()
print('ids =', sortIds)


b = a[sortIds]
print('a sorted:', b)

c = aa[sortIds]
print('aa sorted:\n',c)



