import numpy as np
a=np.array([[ 0.,  5.,  2.],
       [ 0.,  0.,  3.],
       [ 0.,  0.,  0.]])
b=np.array([ 0.,  0.,  3.])
print(np.where(np.all(a==b,axis=1).reshape(a.shape[0],-1)==True))