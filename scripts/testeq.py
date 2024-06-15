import numpy as np


m = np.zeros((5,5)) + 2
print(m)
a = [1,3]


for i in range(5):
    for j in range(5):
        if i!=j and i in a and j in a:
            m[i,j]=0
        elif i>=j and j in a:
            m[i,j]=1
        elif j>a[0] and i>=j:
            m[i,j]=1
        elif j>a[1] and i>=j:
            m[i,j]=1
        else:
            m[i,j]=0

print(m)