import pandas as pd
df = pd.read_csv('student_productivity_distraction_dataset_20000.csv')
print(df)

print(df.isna().sum())
print(df.duplicated().sum())

numerical = ['study_hours_per_day','focus_score','sleep_hours','attendance_percentage']

for col in numerical:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df = df[(df[col]>=lower) & (df[col]<=upper)]

categorical = df['gender']
categorical_dummies = pd.get_dummies(categorical,dtype=float)

x = pd.concat([categorical_dummies,df[numerical]],
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

# SVR (scaled)
svr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR())
])
svr_grid = GridSearchCV(
    svr_pipe,
    {
        "svr__kernel": ["linear", "rbf"],
        "svr__C": [0.01, 0.1, 1, 10,100],
        "svr__gamma": ["scale"]
    },
    cv=5,
    scoring="r2",
    n_jobs=-1
)
svr_grid.fit(x_train, y_train)
best_svr = svr_grid.best_estimator_
print("SVR best params:", svr_grid.best_params_)
print('Train R2:',best_svr.score(x_train,y_train))
print('Test R2:',best_svr.score(x_test,y_test))

# Random Forest Regressor
rfr_grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    {
        "n_estimators": [100,300],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ['sqrt',0.5, 0.7]
    },
    cv=5,
    scoring="r2",
    n_jobs=-1
)
rfr_grid.fit(x_train, y_train)
best_rfr = rfr_grid.best_estimator_
print("RF best params:", rfr_grid.best_params_)
print('Train R2:',best_rfr.score(x_train,y_train))
print('Test R2:',best_rfr.score(x_test,y_test))

