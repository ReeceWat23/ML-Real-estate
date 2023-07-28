import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR



def file_checker(name_of_file):
    while not os.path.exists(name_of_file) is True:
        print("file not found")
        name_of_file = input("Please input a valid file name: ")
    return name_of_file


def read_file(file):
    housing_list = []
    f = open(file, "r")
    for line in f:
        line = line.split(",")
        housing_list.append(line)
    f.close()
    return housing_list


def convert_float(list):
    new_list = []
    for i in list:
        temp = float(i)
        new_list.append(temp)
    return new_list


inventory = pd.read_csv("Inventory.csv")
Unemoloy_rate = pd.read_csv("Unemployment.csv")
Mortgage_rate = pd.read_csv("MortgageRChange.csv")
housing_prices = pd.read_csv("Housing_Prices.csv")

#
# print(inventory.head())
# print(Unemoloy_rate.head())
# print(Mortgage_rate.head())
# print(housing_prices.head()+"\n")


# Convert the columns to Pandas Series
unemployment_rate_change_series = Unemoloy_rate["Unemployment rate change"].to_list()
mortgage_change_series = Mortgage_rate["Mortgage rate change %"].to_list()
prices_change_series = housing_prices['prices change %'].to_list()

Mortgage_severity_list= []

# Define severity
# Decrese or increase over 20% is a very strong change
# Decrease or increase over 10% is a strong change change
# Decrease or increase over 3% is a moderate change
# decrease or increase less than 3 is a slight change
# Decrease or increase less than 1% is a very slight change


# first model is to test severity of the change

for i in mortgage_change_series:
    if float(i) >= 18 or float(i) <= -18:
        Mortgage_severity_list.append("Very strong change")
    elif float(i) >= 8 or float(i) <= -8:
        Mortgage_severity_list.append("Strong change")
    elif float(i) >= 4 or float(i) <= -4:
        Mortgage_severity_list.append("Moderately strong change")
    elif float(i) <= 4 or float(i) >= -4:
        Mortgage_severity_list.append("Slight change ")

# Create a DataFrame from the three lists
df = pd.DataFrame({
    "Unemployment rate change": unemployment_rate_change_series,
    ## data now is checking the change for the next month
    "Mortgage Change severity": Mortgage_severity_list,
    "Housing prices %": prices_change_series

})

df2 = pd.DataFrame({
    "Unemployment rate change": unemployment_rate_change_series,
    "Housing prices %": prices_change_series

})

print(df)

# input data set without the data we want to predict
x = df2

# output data set
y = df["Mortgage Change severity"]

model = RandomForestClassifier(n_estimators=100, max_depth=4)
model.fit(x,y)


# Split the DataFrame into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.25)

# Create a Random Forest regressor

# Fit the model to the training set
model.fit(X_train, y_train)

# Predict the mortgage change for the testing set
predicted_mortgage_changes = model.predict(X_test)

# Print the accuracy of the model
accuracy = model.score(X_test, y_test)
print(accuracy)
# pred = model.predict([[9.25,-0.34737],[22,-2.3]])
#
# print(pred)



