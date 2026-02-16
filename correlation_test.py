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

print(df.select_dtypes(include='number').corr()['productivity_score'].sort_values(ascending=False))