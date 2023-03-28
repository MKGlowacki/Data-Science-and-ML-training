import numpy as np


#1
X = np.arange(-25,25).reshape(10,5)

#2
ones = np.ones(shape = (X.shape[0], 1))

#3
X_1 = np.append(X, ones, axis=1)

#4
w = np.random.rand(X_1.shape[1])

#5
def predict(x, w):
    total_stimulation = np.dot(x,w)
    if(total_stimulation >= 0):
        return 1
    else: 
        return -1


# predict(X_1[0,], w)    

# for x in X_1:
#     print(predict(x, w))   


y = np.array([1, -1, -1, 1, -1, 1, -1, -1, 1, -1])
eta = 0.01


epochs = 10
 
for e in range(epochs):
    for x, y_target in zip(X_1,y):
        y_pred = predict(x, w)
        delta_w = eta * (y_target - y_pred) * x
        w += delta_w
        print(w)