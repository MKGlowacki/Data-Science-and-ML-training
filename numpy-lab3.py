import numpy as np
# 1
x = np.arange(1, 26).reshape(5, 5)

# 2
ones = np.ones(x.shape)

# 3
np.dot(x, ones)

# 4
diag = np.zeros(np.shape(x))

np.fill_diagonal(diag, 1)

np.dot(x, diag)

# 5
np.where(x > 10, 1, 0)
np.where((x % 2) == 0, 1, 0)
np.where((x % 2) == 0, x, x+1)

# 6
x_bis = np.where(x > 10, x*2, 0)

np.count_nonzero(x_bis)

# 7
x = np.array([[10,20,30], [40,50,60]])
y = np.array([[100], [200]])

np.append(x, y, axis=1)

#8

x = np.array([[10,20,30], [40,50,60]])
y = np.array([[100, 200, 300]])

np.append(x,y,axis=0)

#9

np.append(x,x,axis=0)
