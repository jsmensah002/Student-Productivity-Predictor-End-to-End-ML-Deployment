import pandas as pd
df = pd.read_csv('student_productivity_distraction_dataset_20000.csv')
print(df)

print(df.isna().sum())
print(df.duplicated().sum())

numerical = df[['study_hours_per_day','focus_score','sleep_hours','attendance_percentage',
                ]]

categorical = df['gender']
categorical_dummies = pd.get_dummies(categorical,dtype=float)

x = pd.concat([categorical_dummies,numerical],
              axis='columns')

y = df['productivity_score']

print(x)
print(y)

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

linear_reg = LinearRegression()
svr = SVR()
rf = RandomForestRegressor(random_state=42)

linear_reg.fit(x,y)
svr.fit(x,y)
rf.fit(x,y)

from sklearn.model_selection import train_test_split as tt
x_train,x_test,y_train,y_test = tt(x,y,test_size=0.2,random_state=42)

linear_reg.fit(x_train,y_train)
svr.fit(x_train,y_train)
rf.fit(x_train,y_train)

print(linear_reg.score(x_train,y_train))
print(linear_reg.score(x_test,y_test))

print(svr.score(x_train,y_train))
print(svr.score(x_test,y_test))

print(rf.score(x_train,y_train))
print(rf.score(x_test,y_test))


