from sklearn.metrics import  roc_auc_score
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_score,
                                     cross_validate, train_test_split)
import lightgbm as lgbm
import numpy as np


def cross_validation_score_statement(estimator, X, y, scoring, n_splits=5, statement=None, random_state=0):
    """
    Evaluate a score by cross-validation. 
    The fit method will be performed on the entire train subset at each iteration,
    the predict method and scoring will be performed only for objects from test subset where statement is True
    
    Parameters
    ----------
    estimator : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.
    X : pandas.DataFrame
        The data to fit.
    y : pandas.Series
        The target variable to try to predict.
    scoring : callable 
        The scoring function of signature scoring(y_true,y_pred).
    statement : boolean numpy.array of shape equal to y.shape
        The mask showing the objects we want to evaluate estimator on.
    n_splits : int
        Number of folds for cross-validation
    random_state : int
        Random_state for KFold and StratifiedKFold    
    
    Returns
    -----------
    scores : array of float, shape=(n_splits,)
    
    """
    if statement is None:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_iter = list(cv.split(X, y))
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_iter = list(cv.split(X, statement))
    scores = []

    for train, test in cv_iter:
        estimator.fit(X.iloc[train, :].values, y.iloc[train].values)
        if statement is not None:
            y_statement = y.iloc[test].loc[statement[test]]
            pred_statement = estimator.predict_proba(
                X.iloc[test, :].loc[statement[test]].values)[:, 1]
        else:
            y_statement = y.iloc[test]
            pred_statement = estimator.predict_proba(X.iloc[test, :].values)[:, 1]
        scores.append(scoring(y_statement, pred_statement))
    return np.array(scores)

def cross_validate_model(MF_dataset, train_ids, target, lgbm_params, scoring=roc_auc_score, n_splits=5,
                         statement=None, random_seed=424, use_same_params=True):
    '''
    Evaluate model score on each target by cross-validation.
    
    Parameters
    ----------
    MF_dataset : ALS_BPR_Dataset
        Dataset to perform cross validation on.
    train_ids : pd.Series
        Indices of train objects.
    target : pd.DataFrame
        pd.DataFrame of shape (n_train_objects, 5).
    lgbm_params : list of dicts or dict
        If use_same_params is True same LGBM parameters are used for each target.
        If use_same_params is False lgbm_params must be a list of 5 dicts.
    scoring : function from sklearn.metrics
        Metric by which the model would be evaluated.
    n_splits : int
        Amount of splits for cross validation.
    statement : boolean numpy.array of shape equal to y.shape
        The mask showing the objects we want to evaluate estimator on.
    random_seed : int
    use_same_params : bool
    
    Returns
    ----------
    auc_scores : np.array of shape (5, )
        Mean scores for each target.
    '''
    
    if not use_same_params:
        assert len(lgbm_params) == 5, "Amount of parameter dicts must equal to 5"
    else:
        assert isinstance(lgbm_params, dict), "When use_same_parameters is True, lgbm_params must be a dict"
        lgbm_params = 5 * [lgbm_params]
    
    auc_scores = []
    for i, params in enumerate(lgbm_params):
        model = lgbm.sklearn.LGBMClassifier(**params)
        scores = cross_validation_score_statement(estimator=model, X=MF_dataset.datasets[i].iloc[train_ids], y=target['{}'.format(i+1)],
                                                  scoring=scoring ,n_splits=n_splits, statement=statement, random_state=random_seed)
        print("Target {}: mean = {:.4f}, std = {:.4f}".format(i + 1, scores.mean(), np.std(scores)))
        auc_scores.append(scores.mean())

    print("All targets: mean = {:.4f}, std = {:.4f}".format(np.mean(auc_scores), np.std(auc_scores)))
    return auc_scores