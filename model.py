import lightgbm as lgbm
import numpy as np

class LGBM_model():
    '''
    Model with uniqie LGBM Classifier for each target.
    
    Parameters
    ----------
    lgbm_params : list of dicts or dict
        If use_same_params is True same LGBM parameters are used for each target.
        If use_same_params is False lgbm_params must be a list of 5 dicts.
    use_same_params : bool
    '''
    def __init__(self, lgbm_params, use_same_params=True):
        if not use_same_params:
            assert len(lgbm_params) == 5, "Amount of parameter dicts must equal to 5"
            self.lgbm_params = lgbm_params
        else:
            assert isinstance(lgbm_params, dict), "When use_same_parameters is True, lgbm_params must be a dict"
            self.lgbm_params = 5 * [lgbm_params]
    
    def fit(self, MF_dataset, train_ids, target, random_seed=424):
        '''
        Fit an LGBM classifier for each target with parameters specified by self.lgbm_params.
        
        Parameters
        ----------
        MF_dataset : ALS_BPR_dataset
            Dataset with user-item embeddings to fit on.
        target : pd.DataFrame of shape(len(train_ids), 5)
            Target to fit on.
        train_ids: pd.Series
            Ids of objects to train on.
        '''
        self.model_list = []
        for i, params in enumerate(self.lgbm_params):
            params.update({'random_seed':random_seed})
            model = lgbm.sklearn.LGBMClassifier(**params)
            model.fit(MF_dataset.datasets[i].iloc[train_ids], target['{}'.format(i + 1)])
            self.model_list.append(model)
        
    def predict(self, MF_dataset, test_ids):
        '''
        Predict probabilities for each target
        
        Parameters
        ----------
        MF_Dataset : ALS_BPR_Dataset
            Dataset with user-item embeddings to perform predict on.
        test_ids : pd.Series
            Ids of objects to perform predict on.
        
        Returns
        ----------
        predictions : nd.array of shape(len(test_id), 5)
            Predictions
        '''
        predictions_list = []
        for i, model in enumerate(self.model_list):
            predictions = model.predict_proba(MF_dataset.datasets[i].iloc[test_ids])[:, 1]
            predictions_list.append(predictions)
        
        return np.array(predictions_list).T
    
    def fit_predict_n_random_seed(self, MF_dataset, train_ids, target, test_ids, n_random=10):
        '''
        Fit a number of LGBM models with different random_seeds and obtain final predictions
        by averaging predictions of individual models.
        
        Parameters:
        ----------
        MF_Dataset : ALS_BPR_Dataset
            Dataset with user-item embeddings to perform predict on.
        train_ids: pd.Series
            Ids of objects to train on.
        target : pd.DataFrame of shape(len(train_ids), 5)
            Target to fit on.
        test_ids : pd.Series
            Ids of objects to perform predict on.
        n_random : int
            Number of LGBM models to fit on each target.
        
        Returns
        ----------
        predictions : nd.array of shape(len(test_id), 5)
            Predictions
        
        '''
        predictions_list = []
        for i, params in enumerate(self.lgbm_params):
            predictions = 0.0
            for random_seed in range(0, 1000, 1000 // n_random):
                params.update({'random_seed': random_seed})
                model = lgbm.sklearn.LGBMClassifier(**params)
                model.fit(MF_dataset.datasets[i].iloc[train_ids], target['{}'.format(i + 1)])
                predictions += model.predict_proba(MF_dataset.datasets[i].iloc[test_ids])[:, 1]
            predictions_list.append(predictions / n_random)
        
        return np.array(predictions_list).T