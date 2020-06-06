import numpy as np 


x = np.linspace(0,10, 11)

y = [100+i/10 for i in range(5)]

z = np.arange(20).reshape((4,5))

print(x)
print(y)
print(z)


np.savez('testSavezFile.npz', xxx=x, yyy=y, zzz=z)


npzfile = np.load('testSavezFile.npz')

print(npzfile.files)
for dataPart in npzfile.files:
    print(f'data for {dataPart}:')
    print(npzfile[dataPart])