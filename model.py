import pandas as pd 
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
import pickle

data=pd.read_csv('diabetes.csv')

"""
Independent Variables
Number of times pregnant
Plasma glucose concentration a 2 hours in an oral glucose tolerance test
Diastolic blood pressure (mm Hg)
Triceps skin fold thickness (mm)
2-Hour serum insulin (mu U/ml)
Body mass index (weight in kg/(height in m)^2)
Diabetes pedigree function
"""
data.head()
data.dtypes
data.describe()
data.shape
data.keys()
x=data.drop('Outcome',axis=1)
y=data['Outcome']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.4,random_state=520)
model=LogisticRegression()
model.fit(x_train,y_train)
predict=model.predict(x_test)
confusion_matrix(y_test,predict)
accuracy_score(y_test,predict)

pickle.dump(model,open('diabetes_predict.pkl','wb'))

x.keys()