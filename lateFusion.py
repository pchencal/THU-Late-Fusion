import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
import seaborn as sn
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import nltk
from sklearn.metrics import roc_auc_score, f1_score
import seaborn as sn
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error #add rmse
from data import merged
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import r2_score



##### Load data #####
df_train = pd.read_csv('data/merged/merged_cleaned_sentiment_train.csv').drop(['pos','neg','neu', 'compound'], axis = 1)
df_val = pd.read_csv('data/merged/merged_cleaned_sentiment_validation.csv').drop(['pos','neg','neu', 'compound'], axis = 1)
df_test = pd.read_csv('data/merged/merged_cleaned_sentiment_test.csv').drop(['pos','neg','neu', 'compound'], axis = 1)

df_train = df_train[['danceability', 'energy', 'instrumentalness', 'valence','mode', 'y_valence', 'y_arousal']]
df_val = df_val[['danceability', 'energy', 'instrumentalness', 'valence','mode', 'y_valence', 'y_arousal']]
df_test = df_test[['danceability', 'energy', 'instrumentalness', 'valence','mode','y_valence', 'y_arousal']]

df_train = pd.concat([df_train, pd.read_csv('data/lyrics/lyrics_features_train.csv').iloc[:, :-200]], axis=1)
df_val = pd.concat([df_val, pd.read_csv('data/lyrics/lyrics_features_val.csv').iloc[:, :-200]], axis=1)
df_test = pd.concat([df_test, pd.read_csv('data/lyrics/lyrics_features_test.csv').iloc[:, :-200]], axis=1)

df_train = df_train.dropna()
df_val = df_val.dropna()
df_test = df_test.dropna()

print(df_train.columns)



# function to get cross validation scores
def get_cv_scores(model, X_train, y_train):
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    print('CV Mean: ', np.mean(scores))
    print('STD: ', np.std(scores))



### Split data into X and y ###
#     train set
X_train = df_train.drop(['y_valence', 'y_arousal'], axis=1).values
y_train_valence = df_train.y_valence.values 
y_train_arousal = df_train.y_arousal.values
    
#     validation set
X_val = df_val.drop(['y_valence', 'y_arousal'], axis=1).values
y_val_valence = df_val.y_valence.values 
y_val_arousal = df_val.y_arousal.values 

#      test set
X_test = df_test.drop(['y_valence', 'y_arousal'], axis=1).values
y_test_valence = df_test.y_valence.values 
y_test_arousal = df_test.y_arousal.values



###  Linear regression  ###
LinearRegression().fit(y_train_arousal.reshape(-1, 1), 
                       y_train_valence.reshape(-1, 1)).score(y_train_arousal.reshape(-1, 1), y_train_valence.reshape(-1, 1))


def do_regression(X, y_1, y_2, X_validation, y_1_validation, y_2_validation):

    # parameters
    param_grid = {'fit_intercept':[True,False], 'positive':[True, False]}
    
    # Initialize model for Grid search
    lr_val = LinearRegression()
    lr_arou = LinearRegression()
    
    # Grid search
    clf_vale = GridSearchCV(lr_val, param_grid, scoring='r2', verbose=1, n_jobs=-1, return_train_score=True)
    clf_arou = GridSearchCV(lr_arou, param_grid, scoring='r2', verbose=1, n_jobs=-1, return_train_score=True)

    # Print best results on training data    
    clf_vale.fit(X, y_1)
    clf_arou.fit(X, y_2)
    
    # Print best results on training data
    # add new lines to separate rows
    print()
    print("Best parameter for Valence (CV score=%0.3f):" % clf_vale.best_score_)
    print(clf_vale.best_params_)
    
    print()
    print("Best parameter for Arousal (CV score=%0.3f):" % clf_arou.best_score_)
    print(clf_arou.best_params_)
    print()

    #Initialize models with best parameters
    lr_val_top = LinearRegression(fit_intercept=clf_vale.best_params_['fit_intercept'],  positive = clf_vale.best_params_['positive'])
    lr_arou_top = LinearRegression(fit_intercept=clf_arou.best_params_['fit_intercept'], positive = clf_arou.best_params_['positive'])

    # get cross val scores for models 
    get_cv_scores(lr_val_top, X, y_1)
    get_cv_scores(lr_arou_top, X, y_2)

    #fit optimal models to train data 
    lr_val_fit = lr_val_top.fit(X, y_1)
    lr_arou_fit = lr_arou_top.fit(X, y_2)
    
    # validation scores 
    r2_validation_valence = lr_val_fit.score(X_validation, y_1_validation)
    r2_validation_arousal = lr_arou_fit.score(X_validation, y_2_validation)
    
    print()
    print(f'Validation score for Valence: {r2_validation_valence}')
    print(f'Validation score for Arousal: {r2_validation_arousal}')
    
    return clf_vale.best_params_, clf_arou.best_params_



def do_forest_regression(X, y_1, y_2, X_validation, y_1_validation, y_2_validation):
    
    # Initialize models
    rf_val = RandomForestRegressor(random_state=0)
    rf_arou = RandomForestRegressor(random_state=0)
    
    param_grid = {'n_estimators': [100, 500], 'max_depth' : [5,10, 15]}

    # Grid search
    clf_vale = GridSearchCV(rf_val, param_grid, scoring='r2', verbose=1, n_jobs=-1, return_train_score=True)
    clf_arou = GridSearchCV(rf_arou, param_grid, scoring='r2', verbose=1, n_jobs=-1, return_train_score=True)

    # Print best results on training data    
    clf_vale.fit(X, y_1)
    clf_arou.fit(X, y_2)

    # Print best results on training data
    print()
    print("Best parameter for Valence (CV score=%0.3f):" % clf_vale.best_score_)
    print(clf_vale.best_params_)
    
    print()
    print("Best parameter for Arousal (CV score=%0.3f):" % clf_arou.best_score_)
    print(clf_arou.best_params_)

    #Initialize models with best parameters
    rf_val_top = RandomForestRegressor(n_estimators = clf_vale.best_params_['n_estimators'], max_depth = clf_vale.best_params_['max_depth'], random_state=0)
    rf_arou_top = RandomForestRegressor(n_estimators = clf_arou.best_params_['n_estimators'], max_depth = clf_arou.best_params_['max_depth'], random_state=0)

    # get cross val scores
    get_cv_scores(rf_val_top, X, y_1)
    get_cv_scores(rf_arou_top, X, y_2)

    rf_val_fit = rf_val_top.fit(X, y_1)
    rf_arou_fit = rf_arou_top.fit(X, y_2)
    
    r2_validation_valence = rf_val_fit.score(X_validation, y_1_validation)
    r2_validation_arousal = rf_arou_fit.score(X_validation, y_2_validation)

    print()
    print(f'Validation score for Valence: {r2_validation_valence}')
    print(f'Validation score for Arousal: {r2_validation_arousal}')

    return clf_vale.best_params_, clf_arou.best_params_



def do_svr(X, y_1, y_2, X_validation, y_1_validation, y_2_validation):

    # Normalization
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X_validation = scaler.fit_transform(X_validation)

    # Train model
    svr_val = SVR()
    svr_arou = SVR()
    
    param_grid = {'kernel' : ('linear', 'rbf', 'poly'), 'C' : [1,5,10]}

    # Grid search
    clf_vale = GridSearchCV(svr_val, param_grid, scoring='r2', verbose=1, n_jobs=-1, return_train_score=True)
    
    clf_arou = GridSearchCV(svr_arou, param_grid, scoring='r2', verbose=1, n_jobs=-1, return_train_score=True)

    # Print best results on training data    
    clf_vale.fit(X, y_1)
    clf_arou.fit(X, y_2)
    
    # Print best results on training data
    # add new lines to separate rows
    print()
    print("Best parameter for Valence (CV score=%0.3f):" % clf_vale.best_score_)
    print(clf_vale.best_params_)
    
    print()
    print("Best parameter for Arousal (CV score=%0.3f):" % clf_arou.best_score_)
    print(clf_arou.best_params_)
    print()

    # Train model
    svr_val_top = SVR(kernel = clf_vale.best_params_['kernel'], C = clf_vale.best_params_['C'])
    svr_arou_top = SVR(kernel = clf_arou.best_params_['kernel'], C = clf_arou.best_params_['C'])

    # get cross val scores
    get_cv_scores(svr_val_top, X, y_1)
    get_cv_scores(svr_arou_top, X, y_2)

    #fit
    svr_val_fit = svr_val_top.fit(X,y_1)
    svr_arou_fit = svr_arou_top.fit(X, y_2)
    
    r2_validation_valence = svr_val_fit.score(X_validation, y_1_validation)
    r2_validation_arousal = svr_arou_fit.score(X_validation, y_2_validation)
    
    print()
    print(f'Validation score for Valence: {r2_validation_valence}')
    print(f'Validation score for Arousal: {r2_validation_arousal}')

    return clf_vale.best_params_, clf_arou.best_params_



def do_mlp(X, y_1, y_2, X_validation, y_1_validation, y_2_validation):

    # Normalization
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X_validation = scaler.fit_transform(X_validation)

    # Initialize model
    mlp_val = MLPRegressor(random_state = 2)
    mlp_arou = MLPRegressor(random_state = 2)
    
    param_grid = {'hidden_layer_sizes':[(5), (10), (15), (5,5), (10,10), (15,15), (5,5,5), (10,10,10), (15,15,15)], 'max_iter':[500, 1000, 2000, 2500]}

    # Grid search
    clf_vale = GridSearchCV(mlp_val, param_grid, scoring='r2', verbose=1, n_jobs=-1, return_train_score=True)
    clf_arou = GridSearchCV(mlp_arou, param_grid, scoring='r2', verbose=1, n_jobs=-1, return_train_score=True)

    # Print best results on training data    
    clf_vale.fit(X, y_1)
    clf_arou.fit(X, y_2)
    
    # Print best results on training data
    # add new lines to separate rows
    print()
    print("Best parameter for Valence (CV score=%0.3f):" % clf_vale.best_score_)
    print(clf_vale.best_params_)
    
    print()
    print("Best parameter for Arousal (CV score=%0.3f):" % clf_arou.best_score_)
    print(clf_arou.best_params_)
    print()

    #Train model with best params
    mlp_val_top = MLPRegressor(hidden_layer_sizes=clf_vale.best_params_['hidden_layer_sizes'], max_iter=clf_vale.best_params_['max_iter'], random_state = 2)
    mlp_arou_top = MLPRegressor(hidden_layer_sizes=clf_arou.best_params_['hidden_layer_sizes'], max_iter=clf_arou.best_params_['max_iter'], random_state = 2)

    # get cross val scores 
    get_cv_scores(mlp_val_top, X, y_1)
    get_cv_scores(mlp_arou_top, X, y_2)

    mlp_val_fit = mlp_val_top.fit(X, y_1)
    mlp_arou_fit = mlp_arou_top.fit(X, y_2)
    
    r2_validation_valence = mlp_val_fit.score(X_validation, y_1_validation)
    r2_validation_arousal = mlp_arou_fit.score(X_validation, y_2_validation)
    
    print()
    print(f'Validation score for Valence: {r2_validation_valence}')
    print(f'Validation score for Arousal: {r2_validation_arousal}')

    return clf_vale.best_params_, clf_arou.best_params_


# Linear regression
val_par, arou_par = do_regression(X_train, y_train_valence, y_train_arousal, X_val, y_val_valence, y_val_arousal)
lr_val = LinearRegression(fit_intercept=val_par['fit_intercept'], positive = val_par['positive'])
lr_arou = LinearRegression(fit_intercept=arou_par['fit_intercept'], positive = arou_par['positive'])
lr_val_r2 = lr_val.fit(X_train, y_train_valence).score(X_test, y_test_valence)
lr_arou_r2 = lr_arou.fit(X_train, y_train_arousal).score(X_test, y_test_arousal)
lr_val_pred = lr_val.fit(X_train, y_train_valence).predict(X_test)
lr_arou_pred = lr_arou.fit(X_train, y_train_arousal).predict(X_test)
lr_val_rmse = mean_squared_error(y_test_valence, lr_val_pred)
lr_arou_rmse = mean_squared_error(y_test_arousal, lr_arou_pred)

# Random Forest
val_par_rf, arou_par_rf = do_forest_regression(X_train, y_train_valence, y_train_arousal, X_val, y_val_valence, y_val_arousal)
rf_val = RandomForestRegressor(n_estimators = 100, max_depth = 5, random_state=0)
rf_arou = RandomForestRegressor(n_estimators = 100, max_depth = 5, random_state=0)
rf_val_r2 = rf_val.fit(X_train, y_train_valence).score(X_test, y_test_valence)
rf_arou_r2 = rf_arou.fit(X_train, y_train_arousal).score(X_test, y_test_arousal)
rf_val_pred = rf_val.fit(X_train, y_train_valence).predict(X_test)
rf_arou_pred = rf_arou.fit(X_train, y_train_arousal).predict(X_test)
rf_val_rmse = mean_squared_error(y_test_valence, rf_val_pred)
rf_arou_rmse = mean_squared_error(y_test_arousal, rf_arou_pred)

# SVR
val_par_svr, arou_par_svr = do_svr(X_train, y_train_valence, y_train_arousal, X_val, y_val_valence, y_val_arousal)
svr_val = SVR(kernel = val_par_svr['kernel'], C = val_par_svr['C'])
svr_arou = SVR(kernel = arou_par_svr['kernel'], C = arou_par_svr['C'])
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
svr_val_r2 = svr_val.fit(X_train, y_train_valence).score(X_test, y_test_valence)
svr_arou_r2 = svr_arou.fit(X_train, y_train_arousal).score(X_test, y_test_arousal)
svr_val_pred = svr_val.fit(X_train, y_train_valence).predict(X_test)
svr_arou_pred = svr_arou.fit(X_train, y_train_arousal).predict(X_test)
svr_val_rmse = mean_squared_error(y_test_valence, svr_val_pred)
svr_arou_rmse = mean_squared_error(y_test_arousal, svr_arou_pred)

# MLP
val_par_mlp, arou_par_mlp = do_mlp(X_train, y_train_valence, y_train_arousal, X_val, y_val_valence, y_val_arousal)
mlp_val = MLPRegressor(hidden_layer_sizes=val_par_mlp['hidden_layer_sizes'], max_iter=val_par_mlp['max_iter'], random_state = 2)
mlp_arou = MLPRegressor(hidden_layer_sizes=arou_par_mlp['hidden_layer_sizes'], max_iter=arou_par_mlp['max_iter'], random_state = 2)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
mlp_val_r2 = mlp_val.fit(X_train, y_train_valence).score(X_test, y_test_valence)
mlp_arou_r2 = mlp_arou.fit(X_train, y_train_arousal).score(X_test, y_test_arousal)
mlp_val_pred = mlp_val.fit(X_train, y_train_valence).predict(X_test)
mlp_arou_pred = mlp_arou.fit(X_train, y_train_arousal).predict(X_test)
mlp_val_rmse = mean_squared_error(y_test_valence, mlp_val_pred)
mlp_arou_rmse = mean_squared_error(y_test_arousal, mlp_arou_pred)


### Late Fusion Methods ###
#     Simple Averaging 
simple_avg_valence = (lr_arou_pred + rf_val_pred + svr_val_pred + mlp_val_pred) / 4
simple_avg_arousal = (lr_arou_pred + rf_arou_pred + svr_arou_pred + mlp_arou_pred) / 4
print(f'Simple Averaging Valence: {r2_score(y_test_valence, simple_avg_valence)}')
print(f'Simple Averaging Arousal: {r2_score(y_test_arousal, simple_avg_arousal)}')

#     Weighted Averaging
sum_val_reciprocals = 1/ lr_val_rmse + 1/ rf_val_rmse + 1/ svr_val_rmse + 1/ mlp_val_rmse
sum_arou_reciprocals = 1/ lr_arou_rmse + 1/ rf_arou_rmse + 1/ svr_arou_rmse + 1/ mlp_arou_rmse
val_weights = [1/lr_val_rmse/sum_val_reciprocals, 1/rf_val_rmse/sum_val_reciprocals, 1/svr_val_rmse/sum_val_reciprocals, 1/mlp_val_rmse/sum_val_reciprocals]
arou_weights = [1/lr_arou_rmse/sum_arou_reciprocals, 1/rf_arou_rmse/sum_arou_reciprocals, 1/svr_arou_rmse/sum_arou_reciprocals, 1/mlp_arou_rmse/sum_arou_reciprocals]
weighted_avg_valence = np.dot(val_weights, [lr_val_r2, rf_val_r2, svr_val_r2, mlp_val_r2])
weighted_avg_arousal = np.dot(arou_weights, [lr_arou_r2, rf_arou_r2, svr_arou_r2, mlp_arou_r2])
print(f'Weighted Averaging Valence: {weighted_avg_valence}')
print(f'Weighted Averaging Arousal: {weighted_avg_arousal}')


#     Ensemble Learning
base_models = [('lr', LinearRegression()), ('svr', SVR()), ('rf', RandomForestRegressor(random_state=42))]
meta_model =MLPRegressor(random_state=42)
stacking = StackingRegressor(estimators=base_models, final_estimator=meta_model)
stacking_valence = stacking.fit(X_train, y_train_valence).score(X_test, y_test_valence)
stacking_arousal = stacking.fit(X_train, y_train_arousal).score(X_test, y_test_arousal)
print(f'Stacking Valence: {stacking_valence}')
print(f'Stacking Arousal: {stacking_arousal}')

