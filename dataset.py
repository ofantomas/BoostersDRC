import pandas as pd
import numpy as np
import implicit
from scipy import sparse

class ALS_BPR_Dataset():
    '''
    Dataset with user embeddings based on X2 matrix.

    Parameters
    ----------
    ALS_params : dict
        Parameters for ALS method for Implicit Matrix Factorization
    BPR_params : dict
        Parameters for BPR method for Implicit Matrix Factorization
    config : list of string
        Dataset parameters for each target. 'als' -- ALS only, 'bpr' -- BPR only, 'cat' -- concatenation of ALS and BPR
    item_user_emb : list of string
        How to get user embeddings. 'user' -- user embeddings only,
        'item' -- aggregation of item embeddings, 'user_item' -- concatenation
    item_agg : string
        How to aggregate item embeddings. 'mean' -- mean of item embeddings for user,
        'exp_smoothing' -- exponential smoothing
    alpha : float
        Parameter of exponential smoothing

    Attributes
    ----------
    datasets : list of pd.DataFrame
        List of datasets for each target
    '''
    def __init__(self, ALS_params, BPR_params, config, item_user_emb, item_agg='mean', alpha=0.5):
        self.als = implicit.als.AlternatingLeastSquares(**ALS_params)
        self.bpr = implicit.bpr.BayesianPersonalizedRanking(**BPR_params)
        
        self.config = config
        self.item_user_emb = item_user_emb
        
        assert len(self.config) == len(self.item_user_emb), "The length of config must match the lenght of item_user_emb"
        
        self.item_agg = item_agg
        self.alpha = alpha
        
    def fit(self, X, user_item_df):
        '''
        Get item and user embeddings.
        
        Parameters
        ----------
        X: pd.DataFrame
            DataFrame with features from X1
        
        user_item_df: pd.DataFrame
            Dataframe with columns: 'id', 'A'.
            Each row pf df consists of a user_id and an item which the user has 'liked'. 
        '''
        rows, row_pos = np.unique(user_item_df.iloc[:, 0], return_inverse=True)
        cols, col_pos = np.unique(user_item_df.iloc[:, 1], return_inverse=True)
        
        data = np.ones(len(row_pos))
        
        sparse_coo_matrix = sparse.coo_matrix((data, (row_pos, col_pos)), shape=(user_item_df.id.nunique(), user_item_df.A.nunique()))
        
        self.als.fit(sparse.csr_matrix(sparse_coo_matrix).T)
        self.bpr.fit(sparse.csr_matrix(sparse_coo_matrix).T)
        
        ALS_mean_item_embeddings = user_item_df.groupby('id').apply(lambda x: self.get_mean_item_embedding(x, self.als, self.item_agg, self.alpha)).\
                                     reset_index(level=[0, 1]).drop('level_1', axis=1)
        BPR_mean_item_embeddings = user_item_df.groupby('id').apply(lambda x: self.get_mean_item_embedding(x, self.bpr, self.item_agg, self.alpha)).\
                                     reset_index(level=[0, 1]).drop('level_1', axis=1)
        
        self.datasets = []
        for dataset_type, item_user_type in zip(self.config, self.item_user_emb):
            if dataset_type == 'als':
                if item_user_type == 'user':
                    self.datasets.append(self.als.user_factors)
                elif item_user_type == 'item':
                    self.datasets.append(ALS_mean_item_embeddings.iloc[:, 1:].values)
                elif item_user_type == 'user_item':
                    self.datasets.append(np.hstack((self.als.user_factors, ALS_mean_item_embeddings.iloc[:, 1:].values)))
                else:
                    raise AttributeError("Unknown embedding type")
            if dataset_type == 'bpr':
                if item_user_type == 'user':
                    self.datasets.append(self.bpr.user_factors)
                elif item_user_type == 'item':
                    self.datasets.append(BPR_mean_item_embeddings.iloc[:, 1:].values)
                elif item_user_type == 'user_item':
                    self.datasets.append(np.hstack((self.bpr.user_factors, BPR_mean_item_embeddings.iloc[:, 1:].values)))
                else:
                    raise AttributeError("Unknown embedding type")
            if dataset_type == 'cat':
                assert isinstance(item_user_type, dict), "When dataset type is 'cat' item_user_type must be a dict"
                assert set(list(item_user_type.keys())) == set(['als', 'bpr']), "Dictionary must contain specifications for both ALS and BPR"
                
                tmp_feature = np.empty((self.als.user_factors.shape[0], 0))
                
                if item_user_type['als'] == 'user':
                    tmp_feature = np.hstack((tmp_feature, self.als.user_factors))
                elif item_user_type['als'] == 'item':
                    tmp_feature = np.hstack((tmp_feature, ALS_mean_item_embeddings.iloc[:, 1:].values))
                elif item_user_type['als'] == 'user_item':
                    tmp_feature = np.hstack((tmp_feature, self.als.user_factors, ALS_mean_item_embeddings.iloc[:, 1:].values))
                else:
                    raise AttributeError("Unknown embedding type")
                    
                if item_user_type['bpr'] == 'user':
                    tmp_feature = np.hstack((tmp_feature, self.bpr.user_factors))
                elif item_user_type['bpr'] == 'item':
                    tmp_feature = np.hstack((tmp_feature, BPR_mean_item_embeddings.iloc[:, 1:].values))
                elif item_user_type['bpr'] == 'user_item':
                    tmp_feature = np.hstack((tmp_feature, self.bpr.user_factors, BPR_mean_item_embeddings.iloc[:, 1:].values))
                else:
                    raise AttributeError("Unknown embedding type")
                
                self.datasets.append(tmp_feature)

        self.datasets = [pd.concat((X, pd.DataFrame(dataset)), axis=1).drop('id', axis=1) for dataset in self.datasets]
    
    def get_mean_item_embedding(self, s, MF, mode='mean', alpha=0.01):
        '''
        Aggregate embeddings of items, visited by each user, to obtain user embedding.
        
        Parameters
        ----------
        MF : recommender model
            Matrix Factorization model with item_factors_ attribute
        mode : string
            'mean' -- obtain user embedding through averaging of item embeddings,
            'exp_smoothing' - obtain user embedding through exponential smoothing of item embeddings
        alpha : float
            Exponential smoothing parameter
        '''
        if mode == 'mean':
            item_embeddings = MF.item_factors[s.A.values].mean(axis=0)
        elif mode == 'exp_smoothing':
            item_embeddings = [(alpha ** i) * (1 - alpha) * x for i, x in enumerate(MF.item_factors[s.A.values][::-1, :])]
            item_embeddings[-1] /= (1 - alpha)
            item_embeddings = sum(item_embeddings)
        else:
            raise(NotImplemented)
        return pd.DataFrame(data=item_embeddings.reshape(1, -1))