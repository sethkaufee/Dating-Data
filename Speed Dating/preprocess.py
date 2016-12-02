from sklearn.metrics import classification_report, accuracy_score, roc_auc_score,roc_curve,auc
from sklearn.model_selection import StratifiedKFold, train_test_split,cross_val_score
from sklearn.preprocessing import Imputer,StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator
from knnimpute import knn_impute_few_observed
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def split_impute_scale(X, y, scale_data, resample_data, impute_data, test_size):
    
    cols=X.columns
    temp_impute = Imputer(strategy='mean')
    scale = StandardScaler()
    scaler = Pipeline([('imp',temp_impute),('scale',scale)])
    
    if not impute_data:
        X.dropna(inplace=True)
        y = y.loc[X.index]
    
    imputer = KNeighborsImputer(n_neighbors=6)
    xtrain, xtest, ytrain, ytest = train_test_split(X, y,  test_size=test_size, stratify=y, random_state=1000)
    
    train_mask=xtrain.copy().notnull()
    test_mask=xtest.copy().notnull()
    
    steps = []
    if scale_data:
        xtrain[cols] = scaler.fit_transform(xtrain)
        xtest[cols] = scaler.transform(xtest)
        xtrain.where(train_mask,inplace=True)
        xtest.where(test_mask,inplace=True)
        
    if impute_data:
        xtrain = imputer.fit_transform(xtrain)
        xtest = imputer.transform(xtest)
        if resample_data:
            resamp = SMOTE(k=6, m=15)
            xtrain, ytrain = resamp.fit_sample(xtrain, ytrain)           
    
    return xtrain, xtest, ytrain, ytest


class KNeighborsImputer(BaseEstimator,TransformerMixin):
        
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

def modelfit(model, X, y, title='Model', test_size=.25, splits=4, 
             scale_data=True, resample_data=True, impute_data=True):
    
    sns.set_palette(palette='muted')
    xmask = X.copy()
    ymask = y.copy()
    
    xcolumns = X.columns
    
    xtrain, xtest, ytrain, ytest = split_impute_scale(X=xmask, y=ymask, scale_data=scale_data,test_size=test_size,
                                                      resample_data=resample_data, impute_data=impute_data)

    kfold = StratifiedKFold(n_splits=splits,shuffle=True,random_state=8888)
    model.fit(xtrain, ytrain)
    
    predictions = model.predict(xtest)
    probability = model.predict_proba(xtest)[:,1]
    cv_score = cross_val_score(model, xtrain, ytrain, cv=kfold, scoring='recall')
    
    print ("Classification\n")
    print ("Accuracy : {0:.4f}".format(accuracy_score(ytest, predictions)))
    print ("AUC Score (Train): {0:.4f}".format(roc_auc_score(ytest, probability)))
    print ("CV Score : Mean {0:.4f} | Std {1:.4f} | Min {2:.4f} | Max {3:.4f}".format(
            np.mean(cv_score),
            np.std(cv_score),
            np.min(cv_score),
            np.max(cv_score)))
    print(classification_report(ytest, predictions))
    
    if hasattr(model,'feature_importances_'):  
        fig=plt.figure(figsize=(13,5))
        feat_imp = pd.Series(model.feature_importances_, xcolumns).sort_values(ascending=False)
        feat_imp = feat_imp.iloc[:30]
        feat_plot = feat_imp.plot(kind='bar',title='Feature Importances: {}'.format(title),fontsize=10)
        plt.ylabel('Feature Importance')
        fig.autofmt_xdate(bottom=0,rotation=45)
        plt.show()
        return fig
    else:
        try:
            fig=plt.figure(figsize=(13,5))
            coefs_imp = pd.Series(model.coef_[0], xcolumns)
            imask=coefs_imp.abs().sort_values(ascending=False)[:30].index            
            coefs_imp = coefs_imp.loc[imask].sort_values(ascending=False)
            coef_fig = coefs_imp.plot(kind='bar',title='Coef Values: {}'.format(title),fontsize=10)
            plt.ylabel('Coef Values')
            fig.autofmt_xdate(bottom=0,rotation=45)
            plt.show()
            return fig
        except:
            print('NO COEFS')




def sample_plotter(X, y):
    colors = [(0.49803921580314636, 0.78823530673980713, 0.49803921580314636), (0.74219147983719336, 0.68359863477594707, 0.82745099032626435)]    
    size = 10
    a = .6
    v0 = 120
    v1 = 100

    impute = KNeighborsImputer(6)
    X = impute.fit_transform(X)

    sm = SMOTE(k=6, m=15, out_step=0.1)

    xre, yre = sm.fit_sample(X,y)
        
    x_noise = lambda x: np.random.normal(0,.7,size=x.shape[0])
    y_noise = lambda x: np.random.normal(0,50,size=x.shape[0])
    
    dec_dict = y.value_counts().to_dict()
    resamp_dict = pd.Series(yre).value_counts().to_dict()
    
    x0 = np.linspace(v0,v0,dec_dict[0]) + x_noise(np.linspace(v0,v0,dec_dict[0]))
    y0 = np.linspace(1,100,dec_dict[0]) + y_noise(np.linspace(1,100,dec_dict[0]))
    
    no_match0 = dict(
                    x = np.linspace(v0,v0,dec_dict[0]) + x_noise(np.linspace(v0,v0,dec_dict[0])),
                    y = np.linspace(1,100,dec_dict[0]) + y_noise(np.linspace(1,100,dec_dict[0])))
    match0 = dict(
                    x = np.linspace(v1,v1,dec_dict[1]) + x_noise(np.linspace(v1,v1,dec_dict[1])),
                    y = np.linspace(1,100,dec_dict[1]) + y_noise(np.linspace(1,100,dec_dict[1])))
    
    
    no_match1 = dict(
                    x = np.linspace(v0,v0,resamp_dict[0]) + x_noise(np.linspace(v0,v0,resamp_dict[0])),
                    y = np.linspace(1,100,resamp_dict[0]) + y_noise(np.linspace(1,100,resamp_dict[0])))
    match1 = dict(
                    x = np.linspace(v1,v1,resamp_dict[1]) + x_noise(np.linspace(v1,v1,resamp_dict[1])),
                    y = np.linspace(1,100,resamp_dict[1]) + y_noise(np.linspace(1,100,resamp_dict[1])))

    ################ MAKE PLOT #################
    fig = plt.figure()
    ax0 = fig.add_subplot(211)
    ax1 = fig.add_subplot(212)
    
    ax0.scatter(x = no_match0['x'], y = no_match0['y'],c=colors[1],alpha=a,s=size)
    ax0.scatter(x = match0['x'], y = match0['y'],c=colors[0],alpha=a,s=size)

    ax1.scatter(x = no_match1['x'], y = no_match1['y'],c=colors[1],alpha=a,s=size)
    ax1.scatter(x = match1['x'], y = match1['y'],c=colors[0],alpha=a,s=size)
    
    ax0.set_xticklabels(['Yes','No'])
    ax1.set_xticklabels(['Yes','No'])
    
    ax0.set_yticklabels('')
    ax1.set_yticklabels('')
    ax0.set_xticks([v1,v0])
    ax1.set_xticks([v1,v0])
    plt.show()
    return fig,ax0,ax1


def roc_curve_plot(y_true, y_score,title):

    # Compute fpr, tpr, thresholds and roc auc
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.plot(fpr, tpr, label='ROC curve (area = {0:.2f}, {1})'.format(roc_auc, title))
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate or (1 - Specifity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")





def rename_columns(df):
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
    return df.rename(columns=column_changes)






