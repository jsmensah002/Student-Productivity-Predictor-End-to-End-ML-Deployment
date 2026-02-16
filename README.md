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

Results from Not Optimized but scaled:
- LR: Train 80% of data score :0.851551417 || Test 20% of data score : 0.849371568
- SVR: Train 80% of data score :0.84677434 || Test 20% of data score : 0.84198605
- RF: Train 80% of data score :0.97664131 || Test 20% of data score : 0.97675108

Results from Optimized with Outliers:
- LR: Train 80% of data score :0.851551417 || Test 20% of data score : 0.849371568
- SVR: Train 80% of data score :0.85150283 || Test 20% of data score : 0.84926877
- RF: Train 80% of data score :0.89365762 || Test 20% of data score : 0.8430198

Results from Optimized without Outliers:
- LR: Train 80% of data score :0.851551417 || Test 20% of data score : 0.849371568
- SVR: Train 80% of data score :0.84768429 || Test 20% of data score : 0.84762372
- RF: Train 80% of data score :0.89270446 || Test 20% of data score : 0.83959093

Model Selection:
- Based on the results, the Random Forest model from the “Not Optimized but Scaled” experiment initially appears to be the best-performing model due to its high score. However, this performance is misleading. When Random Forest is isolated and evaluated on its own, the accuracy drops, revealing that the apparent improvement was a facade caused by interactions with the other scaled models in the evaluation pipeline.
- This highlights an important point in model evaluation: a model’s high performance within a combined or comparative framework does not always reflect its true standalone predictive power. It underscores the need to test each model independently to validate real performance before concluding it is superior.
-The LR model from the ‘Optimized with Outliers’ experiment was ultimately selected for deployment because of its high accuracy, hyperparameter tuning, and deployability.

Deployment:
- The LR model from the ‘Optimized with Outliers’ experiment was packaged into a Flask app and deployed locally using ngrok.
- Flask: Served the model as an API endpoint.
- Ngrok: Created a secure public URL for real-time predictions without needing a full cloud deployment.

Key Insights:
- This project shows how preprocessing, scaling, and tuning affect model performance. High scores can be misleading if models aren’t tested independently, as with Random Forest.
- Improvements from hyperparameter tuning or outlier removal don’t always guarantee the best model.


