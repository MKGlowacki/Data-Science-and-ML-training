import pandas as pd


iris = pd.read_csv(r"C:\Users\MGlowacki\Desktop\projekty\ML\DANE\iris.data",
                   header = None, 
                   names = ['petal length', 'petal width', 
                            'sepal length', 'sepal width', 'species'])



iris.head()

from sklearn.linear_model import LinearRegression

X = iris.iloc[:, :4]
y = iris.loc[:,'species']


categories ={'Iris-setosa':1, 'Iris-versicolor':2,'Iris-virginica':3}
y = y.apply(lambda x: categories[x])

X.head()
y.head()

lr = LinearRegression()
lr.fit(X.values,y)
lr.score(X.values,y)

iris_1 = [5, 3.5, 1.4, 0.2]
iris_2 = [6.4, 3, 4.5, 1]
iris_3 = [6, 3, 5, 2]
other = [1, 2, 3, 4]
flowers = [iris_1, iris_2, iris_3, other]


species_predict = lr.predict(flowers)
print(species_predict)


for f, s in zip(flowers, species_predict):
    if round(s) == 1:
        print('Flower {} is {}'.format(f, 'Iris-setosa'))
    elif round(s) == 2:
        print('Flower {} is {}'.format(f, 'Iris-versicolor'))
    elif round(s) == 3:
        print('Flower {} is {}'.format(f, 'Iris-virginica'))
    else:
        print('Flower {} is {}'.format(f, 'UNKNOWN'))