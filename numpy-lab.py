import numpy as np

#2
a = np.arange(20)

#3
a.shape

#4
a[0]

a[3]

#5
a = a.reshape(2,10)
a.shape
a

#6
a[0]

#8
a[0][3]

#9
a = a.reshape(2,5,2)

#10
a.shape

a

#11
a[0]
a[0][3]
a[0][3][1]

#12

b = np.arange(0, 40, 2).reshape(4,5)

#13

a_python_list = [2**x for x in range(10)]

c = a_python_list

#14

zero_array = np.zeros(10)

one_array = np.ones(10)

empty_array = np.empty(100)

lucky_array = np.full((5,5), 13)

diagonal_array = np.eye(5)

random_array = np.random.random(10)

linspace_array = np.linspace(100, 200, 5)
