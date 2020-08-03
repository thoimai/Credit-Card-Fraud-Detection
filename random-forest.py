# Import the random forest from sklearn libray
from sklearn.ensemble import RandomForestClassifier

pipeline_rf = Pipeline([
    ('model', RandomForestClassifier(n_jobs=-1, random_state=1))
])


param_grid_rf = {'model__n_estimators': [75]}


grid_rf = GridSearchCV(estimator=pipeline_rf, param_grid=param_grid_rf, scoring=MCC_scorer, 
                            n_jobs=-1, pre_dispatch='2*n_jobs', cv=5, verbose=1, return_train_score=False)

# Perform the grid search --- traning the random forest model 
grid_rf.fit(X_train, y_train)

# Best training score 
grid_rf.best_score_

# Show the optimal parameter
grid_rf.best_params_

