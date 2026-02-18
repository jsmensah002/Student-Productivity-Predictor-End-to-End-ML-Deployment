import pandas as pd
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('student_productivity_distraction_dataset_20000.csv')
print(df)

print(df.isna().sum())
print(df.duplicated().sum())

numerical = df[['study_hours_per_day','focus_score','sleep_hours','attendance_percentage']]

label_encoders = {}
for col in ['gender']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save for future predictions

categorical = df['gender']

x = pd.concat([categorical,numerical],
              axis='columns')

y = df['productivity_score']

print(x)
print(y)

#Importing models/ best params/ best scores
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("\n=== REGRESSION MODELS ===\n")

# Linear Regression
reg_pipe = Pipeline([
    ('scaler',StandardScaler()),
    ('reg',LinearRegression())
])

reg_grid = GridSearchCV(
    reg_pipe,
    {},
    cv=5,
    scoring='r2',
    n_jobs=-1
)
reg_grid.fit(x_train,y_train)

print("Linear Regression best params:",reg_grid.best_params_)
print('Best CV R2:',reg_grid.best_score_)  

best_reg = reg_grid.best_estimator_
print('Train R2:',best_reg.score(x_train,y_train))
print('Test R2:',best_reg.score(x_test,y_test))

import joblib
joblib.dump(best_reg,'final_reg_model.pkl')
model = joblib.load('final_reg_model.pkl')

import joblib
joblib.dump(x.columns, 'columns.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')  # save encoders for Flask