import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

auto = pd.read_csv(r"C:\Users\MGlowacki\Desktop\projekty\ML\DANE\auto-mpg.csv")

auto.head()

X = auto.iloc[:, 1:-1]
X = X.drop('horsepower', axis=1)
y = auto.loc[:,'mpg']


X.head()
y.head()

print(X.to_numpy())

lr = LinearRegression()
lr.fit(X.to_numpy(), y)
lr.score(X.to_numpy(), y)


my_car1 = [4, 160, 190, 12, 90, 1]
my_car2 = [4, 200, 260, 15, 83, 1]
 
cars = [my_car1, my_car2]

mpg_predict = lr.predict(cars)
print(mpg_predict)