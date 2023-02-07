#loading common data related modules

import numpy as np
import pandas as pd
import math

#loading modelling algorithms
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor

#loading tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

#loading visualisation modules 
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

#ignore warning messages
import warnings
warnings.filterwarnings('ignore')

#read data
diamonds = pd.read_csv(r'C:\Users\MGlowacki\Desktop\projekty\ML\DANE\diamonds.csv')

#review and clean the data
#remove unnecessary columns
diamonds.head()
diamonds.drop(['Unnamed: 0'], axis=1, inplace=True)
diamonds.head()

#review the data and get intuition about it
diamonds.shape
diamonds.info()

#find and eliminate nulls
diamonds.isnull().sum()
msno.matrix(diamonds, figsize=(10,4))

#search for illogical values
diamonds.describe()
diamonds.loc[(diamonds['x']==0) | (diamonds['y']==0) | (diamonds['z']==0)]
len(diamonds.loc[(diamonds['x']==0) | (diamonds['y']==0) | (diamonds['z']==0)])

diamonds =diamonds[(diamonds[['x','y','z']] != 0).all(axis=1)]
#check after execution
diamonds.loc[(diamonds['x']==0) | (diamonds['y']==0) | (diamonds['z']==0)]

# Detect dependencies in the data
corr = diamonds.corr()
sns.heatmap(data=corr, square=True, annot=True, cbar=True)
sns.pairplot(diamonds)

#check distribution
sns.kdeplot(diamonds['carat'], shade=True, color='r')
plt.hist(diamonds['carat'], bins=25)

#check correlation graph
sns.jointplot(x='carat', y='price', data=diamonds, size=5)


#analyze feature by feature, create hypotesis, try to find evidence
#sns.factorplot(x='cut', data=diamonds, kind='count', aspect=1.5)
#sns.factorplot(x='cut', y='price', data=diamonds, kind='box', aspect=1.5)

















