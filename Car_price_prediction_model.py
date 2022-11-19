#This dataset contains information about used cars.
#This data can be used for a lot of purposes such as price prediction to exemplify the use of linear regression in Machine Learning.
#The columns in the given dataset are as follows:

#name
#year
#selling_price
#km_driven
#fuel
#seller_type
#transmission
#Owner
#For used motorcycle datasets please go to https://www.kaggle.com/nehalbirla/motorcycle-dataset
#same method we will follow that we was follow in some previous data 
#car data
#data pre-processing
#train-test- split
#In this time will use liner model that is lasso regression model
#----------------------------------------import useful labrary
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics 
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

#------------------------------------------data pre-processing
data = pd.read_csv("C:/Users/kunde/all vs code/ml prject/car data.csv")
print(data.columns)
print(data.shape)
print(data.head(5))
print(data.info())
print(data.isnull().sum())#this is finding missing values in each columns 
print(data["Seller_Type"].unique())

#-------------------------------------------you have to convert this string data into numerical(beacuse machine's are more friendly with numerical than string but string also valid )

print(data.Car_Name.value_counts())
print(data.Fuel_Type.value_counts())
print(data.Seller_Type.value_counts())
print(data.Transmission.value_counts())
data.replace({'Fuel_Type':{'Petrol':0, "Diesel":1, "CNG":2}}, inplace=True)
data.replace({'Transmission':{"Manual":0, "Automatic":1}}, inplace=True)
data.replace({'Seller_Type':{"Dealer":0, "Individual":1}}, inplace=True)
print(data.head(10))
x = data.drop(columns=["Selling_Price", "Car_Name"], axis=1)
y = data["Selling_Price"]
print(x.shape)
print(y.shape)
print(x.columns)
print(y)

#----------------------------------------train test spilt 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
print(x.shape, x_test.shape, x_train.shape)
print(y.shape, y_train.shape, y_test.shape)

#---------------------------------------------use model
model = LinearRegression()
model.fit(x_train, y_train)
#-------------------------------------predictions

x_predict_tr = model.predict(x_train)
print(y_train,"this is true predictions", x_predict_tr, "this is our values")
accu = metrics.r2_score(y_train, x_predict_tr)
print(accu,"this is our model accurancy")

#-------------------------------------test data prediction

x_pred_test = model.predict(x_test)
print(y_test, "this is our true predict values", x_pred_test, "this is our values")
accur = metrics.r2_score(y_test, x_pred_test)
print(accur)
print("----------------------")
#------------------------------new data predictions
print(y[0])
print(x_test.shape)
#-------------------------plot 
plt.scatter(x_predict_tr, y_train)
plt.xlabel("Predictions of our model", fontsize=(25))
plt.ylabel("this is true value", fontsize=(25))
plt.title("THIS IS COMPARISON")
plt.show()



#lesso regression 

model = Lasso()
model.fit(x_train, y_train)
x_predict_tr = model.predict(x_train)
print(y_train,"this is true predictions", x_predict_tr, "this is our values")
accu = metrics.r2_score(y_train, x_predict_tr)
print(accu,"this is our model accurancy")
x_pred_test = model.predict(x_test)
print(y_test, "this is our true predict values", x_pred_test, "this is our values")
accur = metrics.r2_score(y_test, x_pred_test)
print(accur)