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



column_changes={
'imprace'      : 'initial_race_importance',
'imprelig'     : 'initial_religion_importance',
'exphappy'     : 'initial_happy_expectation',
'expnum'       : 'initial_number_expectation',
'attr1_1'      : 'initial_attractive_importance',
'sinc1_1'      : 'initial_sincere_importance',
'intel1_1'     : 'initial_intelligence_importance',
'fun1_1'       : 'initial_fun_importance',
'amb1_1'       : 'initial_ambitious_importance',
'shar1_1'      : 'initial_shared_interests_importance',
'attr4_1'      : 'initial_attractive_same_importance',
'sinc4_1'      : 'initial_sincere_same_importance',
'intel4_1'     : 'initial_intelligence_same_importance',
'fun4_1'       : 'initial_fun_same_importance',
'amb4_1'       : 'initial_ambitious_same_importance',
'shar4_1'      : 'initial_shared_same_importance',
'attr2_1'      : 'initial_attractive_opposite_importance',
'sinc2_1'      : 'initial_sincere_opposite_importance',
'intel2_1'     : 'initial_intelligence_opposite_importance',
'fun2_1'       : 'initial_fun_opposite_importance',
'amb2_1'       : 'initial_ambitious_opposite_importance',
'shar2_1'      : 'initial_shared_opposite_importance',
'attr3_1'      : 'initial_attractive_self_measure',
'sinc3_1'      : 'initial_sincere_self_measure',
'intel3_1'     : 'initial_intelligence_self_measure',
'fun3_1'       : 'initial_fun_self_measure',
'amb3_1'       : 'initial_ambitious_self_measure',
'shar3_1'      : 'initial_shared_self_measure',
'attr5_1'      : 'initial_attractive_others_measure',
'sinc5_1'      : 'initial_sincere_others_measure',
'intel5_1'     : 'initial_intelligence_others_measure',
'fun5_1'       : 'initial_fun_others_measure',
'amb5_1'       : 'initial_ambitious_others_measure',
'shar5_1'      : 'initial_shared_others_measure',                
'attr1_s'      : 'halfway_attractive_importance',
'sinc1_s'      : 'halfway_sincere_importance',
'intel1_s'     : 'halfway_intelligence_importance',
'fun1_s'       : 'halfway_fun_importance',
'amb1_s'       : 'halfway_ambitious_importance',
'shar1_s'      : 'halfway_shared_interests_importance',
'attr3_s'      : 'halfway_attractive_self_measure',
'sinc3_s'      : 'halfway_sincere_self_measure',
'intel3_s'     : 'halfway_intelligence_self_measure',
'fun3_s'       : 'halfway_fun_self_measure',
'amb3_s'       : 'halfway_ambitious_self_measure',
'shar3_s'      : 'halfway_shared_interests_self_measure',
'satis_2'      : 'followup_overall_satisfaction',
'length'       : 'followup_length_dates_satisfaction',
'numdat_2'     : 'followup_num_dates_satisfaction',
'attr7_2'      : 'followup_attractive_influence',
'sinc7_2'      : 'followup_sincere_influence',
'intel7_2'     : 'followup_intelligence_influence',
'fun7_2'       : 'followup_fun_influence',
'amb7_2'       : 'followup_ambitious_influence',
'shar7_2'      : 'followup_shared_interests_influence',
'attr1_2'      : 'followup_attractive_importance',
'sinc1_2'      : 'followup_sincere_importance',
'intel1_2'     : 'followup_intelligence_importance',
'fun1_2'       : 'followup_fun_importance',
'amb1_2'       : 'followup_ambitious_importance',
'shar1_2'      : 'followup_shared_interests_importance',
'attr4_2'      : 'followup_attractive_same_importance',
'sinc4_2'      : 'followup_sincere_same_importance',
'intel4_2'     : 'followup_intelligence_same_importance',
'fun4_2'       : 'followup_fun_same_importance',
'amb4_2'       : 'followup_ambitious_same_importance',
'shar4_2'      : 'followup_shared_same_importance',
'attr2_2'      : 'followup_attractive_opposite_importance',
'sinc2_2'      : 'followup_sincere_opposite_importance',
'intel2_2'     : 'followup_intelligence_opposite_importance',
'fun2_2'       : 'followup_fun_opposite_importance',
'amb2_2'       : 'followup_ambitious_opposite_importance',
'shar2_2'      : 'followup_shared_opposite_importance',
'attr3_2'      : 'followup_attractive_self_measure',
'sinc3_2'      : 'followup_sincere_self_measure',
'intel3_2'     : 'followup_intelligence_self_measure',
'fun3_2'       : 'followup_fun_self_measure',
'amb3_2'       : 'followup_ambitious_self_measure',
'shar3_2'      : 'followup_shared_self_measure',
'attr5_2'      : 'followup_attractive_others_measure',
'sinc5_2'      : 'followup_sincere_others_measure',
'intel5_2'     : 'followup_intelligence_others_measure',
'fun5_2'       : 'followup_fun_others_measure',
'amb5_2'       : 'followup_ambitious_others_measure',
'shar5_2'      : 'followup_shared_others_measure',
'you_call'     : 'final_you_call_count',
'them_cal'     : 'final_them_call_count',
'date_3'       : 'final_match_dates',
'numdat_3'     : 'final_num_match_seen',
'num_in_3'     : 'final_num_match_dates',
'attr1_3'      : 'final_attractive_importance',
'sinc1_3'      : 'final_sincere_importance',
'intel1_3'     : 'final_intelligence_importance',
'fun1_3'       : 'final_fun_importance',
'amb1_3'       : 'final_ambitious_importance',
'shar1_3'      : 'final_shared_interests_importance',
'attr7_3'      : 'final_attractive_influence',
'sinc7_3'      : 'final_sincere_influence',
'intel7_3'     : 'final_intelligence_influence',
'fun7_3'       : 'final_fun_influence',
'amb7_3'       : 'final_ambitious_influence',
'shar7_3'      : 'final_shared_interests_influence',
'attr4_3'      : 'final_attractive_same_importance',
'sinc4_3'      : 'final_sincere_same_importance',
'intel4_3'     : 'final_intelligence_same_importance',
'fun4_3'       : 'final_fun_same_importance',
'amb4_3'       : 'final_ambitious_same_importance',
'shar4_3'      : 'final_shared_same_importance',
'attr2_3'      : 'final_attractive_opposite_importance',
'sinc2_3'      : 'final_sincere_opposite_importance',
'intel2_3'     : 'final_intelligence_opposite_importance',
'fun2_3'       : 'final_fun_opposite_importance',
'amb2_3'       : 'final_ambitious_opposite_importance',
'shar2_3'      : 'final_shared_opposite_importance',
'attr3_3'      : 'final_attractive_self_measure',
'sinc3_3'      : 'final_sincere_self_measure',
'intel3_3'     : 'final_intelligence_self_measure',
'fun3_3'       : 'final_fun_self_measure',
'amb3_3'       : 'final_ambitious_self_measure',
'shar3_3'      : 'final_shared_self_measure',
'attr5_3'      : 'final_attractive_others_measure',
'sinc5_3'      : 'final_sincere_others_measure',
'intel5_3'     : 'final_intelligence_others_measure',
'fun5_3'       : 'final_fun_others_measure',
'amb5_3'       : 'final_ambitious_others_measure',
'shar5_3'      : 'final_shared_others_measure',
}

def rename_columns(df):
    return df.rename(columns=column_changes)






