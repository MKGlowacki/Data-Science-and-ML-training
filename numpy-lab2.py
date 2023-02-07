import numpy as np

#2
arr = np.arange(5, 30, 2)

#3
boolArr = arr < 10

#4
newArr = arr[boolArr]
newArr

newArr = arr[arr < 20]
newArr

newArr = arr[arr%3 == 0]
newArr

newArr = arr[(arr>10) & (arr<20)]
newArr

#5
arr = np.arange(24).reshape(4,6)

arr

arr[0]
arr[0][1]
arr[0, 1:3]
arr[0, 1:4]
arr[0, :]
arr[:, 1]
arr[0:2, 1]
arr[:2, 1]
arr[:2, 1:3]
arr[:,-1]
arr[:, :-1]

#6

arr = np.arange(50).reshape(10,5)

split_level = 0.2
num_rows = arr.shape[0]
split_border = split_level * num_rows

test = arr[:round(split_border), :]
trening = arr[round(split_border):, :]

np.random.shuffle(arr)

test = arr[:round(split_border), :]
trening = arr[round(split_border):, :]

#7

data = np.arange(500).reshape(100,5)

np.random.shuffle(data)

split_level = 0.2
num_rows = data.shape[0]
split_border = round(split_level * num_rows)

X_test = data[:split_border, :-1]
X_train = data[split_border:, :-1]
y_test = data[:split_border, -1]
y_train = data[split_border:, -1]

from sklearn.model_selection import train_test_split
 
X_train, X_test, y_train, y_test = train_test_split(
        data[:, :-1], data[:, -1], test_size=0.2, shuffle = True)

