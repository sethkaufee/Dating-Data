from column_chages import rename_columns_helper
from sklearn.preprocessing import Imputer, RobustScaler,normalize,StandardScaler
from time import time
from knnimpute import knn_impute_few_observed
from sklearn.base import TransformerMixin
import numpy as np
from sklearn.pipeline import Pipeline
import pandas as pd


def categorical_switch(df, predictors):
	for col in predictors:
		if hasattr(df[col],'cat'):
			df[col] = df[col].cat.codes
		elif hasattr(df[col],'str'):
			df[col] = df[col].astype('category').cat.codes
	return df

def wave_scale(dataframe, predictors):
	df = dataframe.copy()
	df = categorical_switch(df, predictors)

	wave_dependent_columns=r"(^initial_((?!race)(?!religion)([a-z]+))(_interests)?_importance|^initial_((?!race)(?!religion)([a-z]+))(_interests)?_same_importance)"    
	wave6_10_index = df.wave.isin(range(6,11))
	
	wave_dependent = predictors[predictors.str.contains(wave_dependent_columns)]
	non_wave_dependent = predictors[~predictors.str.contains(wave_dependent_columns)]

	impute, rscale = Imputer(strategy='most_frequent'), RobustScaler()

	df.loc[wave6_10_index,wave_dependent] = impute.fit_transform(df.loc[wave6_10_index,wave_dependent])
	df.loc[~wave6_10_index,wave_dependent] = impute.fit_transform(df.loc[~wave6_10_index,wave_dependent])
	df.loc[:,non_wave_dependent] = impute.fit_transform(df.loc[:,non_wave_dependent])
	
	df = uneven(df,predictors)

	df.loc[wave6_10_index,wave_dependent] = rscale.fit_transform(df.loc[wave6_10_index,wave_dependent])
	df.loc[~wave6_10_index,wave_dependent] = rscale.fit_transform(df.loc[~wave6_10_index,wave_dependent])
	df.loc[:,non_wave_dependent] = rscale.fit_transform(df.loc[:,non_wave_dependent])

	return df

def uneven(df, predictors):
    iid_dec_df = df.groupby('iid')['dec'].sum()
    pid_dec_df = df.groupby('pid')['dec'].sum()
    
    iid_dec_value_counts = iid_dec_df.value_counts()
    iid_dec_df.map(iid_dec_value_counts)
    
    pid_dec_value_counts = pid_dec_df.value_counts()
    pid_dec_df.map(pid_dec_value_counts)
    
    # maps back into original frame
    bias_multi_factor = df.iid.map(iid_dec_df.map(iid_dec_value_counts))
    unbias_multi_factor = df.pid.map(pid_dec_df.map(pid_dec_value_counts))
    
    # makes a PMF
    df[predictors] = normalize(df[predictors])
    
    # Multiplies the weight by the probability 
    df[predictors] = df[predictors].multiply(bias_multi_factor,axis='index').multiply(1/unbias_multi_factor,axis='index')
    return df




class KNeighborsImputer(TransformerMixin):
        
    """
    Convert missing values with k nearest neighbors
    """
    
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
        
    def transform(self, X, **transform_params):
        
        if hasattr(X, 'columns'):
            columns = X.columns
            index = X.index
            missing_mask = X.isnull().as_matrix()
            X = X.as_matrix()
            
        else:
            missing_mask = np.isnan(X)
                    
        filled_mask = knn_impute_few_observed(X, missing_mask , self.n_neighbors)
        
        failed_to_impute = np.isnan(filled_mask)
        n_missing_after_imputation = failed_to_impute.sum()
        if n_missing_after_imputation != 0:
            print("[KNN] Warning: %d/%d still missing after imputation, replacing with 0" % (
                n_missing_after_imputation,
                X.shape[0] * X.shape[1]))
            filled_mask[failed_to_impute] = X[failed_to_impute]
            
        return pd.DataFrame(data=filled_mask, index=index, columns=columns)

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}







