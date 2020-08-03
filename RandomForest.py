# Import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

# Pipeline for model 
pipeline_rf = Pipeline([
    ('model', RandomForestClassifier(n_jobs=-1, random_state=1))
])


# Implement grid search to try different hyperparameter 
param_grid_rf = {'model__n_estimators': [75]}
grid_rf = GridSearchCV(estimator=pipeline_rf, param_grid=param_grid_rf, scoring=MCC_scorer,
n_jobs=-1, pre_dispatch='2*n_jobs', cv=5, verbose=1, return_train_score=False)

# Run grid search 
grid_rf.fit(X_train, y_train)

# Show best score ~ 0.86 MCC_scorer
grid_rf.best_score_


