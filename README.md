Brief Overview:
- This project explores how data preprocessing and model optimization impact performance, with four experimental setups:
- Not Optimized: Baseline models trained on raw data without any hyperparameter tuning.
- Not Optimized but Scaled: Baseline models trained on scaled features to observe the effect of normalization.
- Optimized with Outliers: Models tuned with hyperparameter optimization including all data, keeping outliers.
- Optimized without Outliers: Models tuned after removing outliers, showing the impact of noise reduction on performance.

Method:
- Studied correlation between inputs and output to validate feature relevance
- Observed how scaling, outlier removal, and tuning affect each model differently
- Models used were Linear Regression (LR), Random Forest Regressor (RF), and Support Vector Regression (SVR)

Results from Not Optimized:
- LR: Train 80% of data score :0.851551417 || Test 20% of data score : 0.849371568
- SVR: Train 80% of data score :0.799348062 || Test 20% of data score : 0.794920043
- RF: Train 80% of data score :0.976657911 || Test 20% of data score : 0.83402173

Results from Not Optimized but Scaled:
- LR: Train 80% of data score :0.851551417 || Test 20% of data score : 0.849371568
- SVR: Train 80% of data score :0.84677434 || Test 20% of data score : 0.84198605
- RF: Train 80% of data score :0.97664131 || Test 20% of data score : 0.97675108
- The features of Linear Regression and SVR were scaled, but RF's features weren't scaled, yet RF’s test score jumped unrealistically. This does not reflect RF’s true predictive ability.

Results from Optimized with Outliers Present:
- LR: Train 80% of data score :0.851551417 || Test 20% of data score : 0.849371568
- SVR: Train 80% of data score :0.85150283 || Test 20% of data score : 0.84926877
- RF: Train 80% of data score :0.89365762 || Test 20% of data score : 0.8430198

Results from Optimized with Outliers Removed:
- LR: Train 80% of data score :0.851551417 || Test 20% of data score : 0.849371568
- SVR: Train 80% of data score :0.84768429 || Test 20% of data score : 0.84762372
- RF: Train 80% of data score :0.89270446 || Test 20% of data score : 0.83959093

Model Selection:
- Based on the results, the Random Forest model from the “Not Optimized but Scaled” experiment appears to be the best-performing model due to its high score. This misleading performance was caused by interactions with the other scaled models in the evaluation pipeline.
- This highlights an important point in model evaluation: a model’s high performance within a combined or comparative framework does not always reflect its true standalone predictive power. It underscores the need to test each model independently to validate real performance before concluding it is superior.
-The LR model from the ‘Optimized with Outliers’ experiment was ultimately selected for deployment because of its high accuracy, hyperparameter tuning, and deployability.

Deployment:
- The LR model from the ‘Optimized with Outliers’ experiment was packaged into a Flask app and deployed using ngrok.
- Ngrok created a secure public URL for real-time predictions without needing a full cloud deployment.

Key Insights:
- This project shows how preprocessing, scaling, and tuning affect model performance.
- When scaling models, exclude any model you don’t intend to scale from the ones being scaled. In the 'Not Optimized but Scaled' experiment, Linear Regression and SVR were scaled but RF was not, yet RF’s test score jumped unrealistically and gave a misleading impression of its true performance. 
- Improvements from hyperparameter tuning or outlier removal don’t always guarantee the best model.


