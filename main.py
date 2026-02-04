# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#load datasets
df = pd.read_csv(r"C:\Users\lalit\Desktop\studentdataset\WA_Fn-UseC_-HR-Employee-Attrition.csv")
print(df.head())

#dataset shape
print(df.shape)
# Dataset information
df.info()

#check missing values
print(df.isnull().sum())
#duplicate value
print(df.duplicated().sum())
#target varible distribution 
print(df['Attrition'].value_counts())
# summary static
print(df.describe())

#visulazition
#countplot
sns.countplot(x='Attrition',data=df)
plt.title("Employee Attrition Count")
plt.xlabel("Attrition")
plt.ylabel("Number of Employees")
plt.show()

#data preprocesing
df['Attrition']= df['Attrition'].map({'yes':1, 'no':0})
#encoded varible
df_encoded = pd.get_dummies(df, drop_first=True)
#featureselect
X=df_encoded.drop('Attrition',axis=1)
y=df_encoded['Attrition']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)