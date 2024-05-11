import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from ctypes import LibraryLoader
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.tree import DecisionTreeRegressor

df = sns.load_dataset('mpg')
df.isnull().sum()
df.dropna(inplace=True)
X = df[['displacement','horsepower','weight','acceleration']]
Y = df.mpg
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.15,random_state=42)

model = LinearRegression()
model.fit(X,Y)
model.score(X_test,Y_test)
model2=DecisionTreeRegressor(criterion='poisson',random_state=0)
model2.fit(X_train,Y_train)
model2.score(X_test,Y_test)

filename='mpg_regression.sav'
pickle.dump(model,open(filename,'wb'))
X_test.loc[0]
loaded_model = pickle.load(open('mpg_regression.sav','rb'))
loaded_model.score(X_test,Y_test)