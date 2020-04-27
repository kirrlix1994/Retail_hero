# usefull funs for uplift model

import gc
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split 

from catboost import CatBoostClassifier
from sklift.metrics import uplift_at_k


def catb_get_feature_imp(catb_est, n_top=20):
    """
    params:
    --------
    catb_est:      catboost model estimator
    n_top:         ccount features to lay plot

    --------
    returns:
        df with feture and its importace value in descending order 
    """
    df_imp = (
        pd.DataFrame({
            "feat": catb_est.get_feature_names(),
            "value": catb_est.get_feature_importance()
        })
        .sort_values("value", ascending=False)
        .head(n_top)
    )

    return df_imp


def feat_imp(estimator, imp_type="gain", n_top=100, top=True, pipeline_flg=True):
    '''
    xgb - model get feature importance
    '''
    if pipeline_flg:
        imp = pd.DataFrame(
            [estimator.get_booster().get_score(importance_type=imp_type)]
        ).T.reset_index()
    imp.columns = ["feature", "value"]
    if top:
        imp = imp.sort_values("value", ascending=False)
    else:
        imp = imp.sort_values("value", ascending=True)

    return imp.head(n_top)


def boosting_feat_imp(
    estimator,
    model,
    n_top=20,
    imp_type='gain',
    top=True
):
    """
    params:
    --------
    estimator:     xgboost / catboost fitted estimator
    model:         'xgb' or 'catb'
    n_top:         count, first features
    imp_type:      for 'xgboost' model only type to order by features
    top:           True / False, if True n_top most important features shown, else bottom
    
    --------
    returns:
        df with feature and its importace value in descending order
        
    """
    if model not in {'xgb', 'catb'}:
        raise ValueError('Wrong model parametr, has to be one of "xgb" or "catb"')
    
    if model == 'xgb':
        df_imp = pd.DataFrame([
            estimator\
                .get_booster()\
                .get_score(importance_type=imp_type)
        ]
        ).T.reset_index()
        df_imp.columns =\
            ["feature", "value"]
            
    elif model == 'catb':
        df_imp = pd.DataFrame({
            "feature": estimator.feature_names_,
            "value": estimator.get_feature_importance()
        })
        
    top_inverse = not top
    
    df_imp = df_imp\
        .sort_values('value', ascending=top_inverse)
            
    return df_imp.head(n_top)



def uplift_at_alpha_orig(y_true, uplift, treatment, alpha=0.1):
    """
    Calculate uplift per percentile
    
    # todo: 
    # - check length equal
    # - warning when cannot sort / thr_int is very low
    """
    y_true, uplift, treatment =\
        np.array(y_true), np.array(uplift), np.array(treatment)
    desc_score_indices =\
        np.argsort(uplift, kind="mergesort")[::-1]
    y_true, uplift, treatment =\
        y_true[desc_score_indices], uplift[desc_score_indices], treatment[desc_score_indices]
    
    thr_int = int(len(y_true)*alpha) + 1
    
    lift_trmnt = np.mean(
        y_true[:thr_int][treatment[:thr_int] == 1]
    )
    
    lift_ctrl = np.mean(
        y_true[:thr_int][treatment[:thr_int] == 0]
    )
    
    return lift_trmnt - lift_ctrl


def get_category_feat(
    df_full, 
    column,
    id_col='client_id',
    agg_col='purchase_sum'
):
    '''
    Get features as aggregates by id_col and column - column with categories
    
    parms:
    ------
    df_full:    pd data frame with id_col and agg_col
    column:     object type column
    id_col:     id, key in resulted df
    agg_col:    col to calculate stats on
    
    result:
    ----
        pd data frame with features
    '''
    
    df_tmp = df_full\
        .groupby([id_col] + [column])\
        .agg({
             id_col: 'count',
             agg_col: 'sum',
        })\
        .rename(columns={
            id_col: 'count'
        })\
        .reset_index()

    df_tmp_2 = df_tmp\
        .sort_values([id_col, 'count'], ascending=False)\
        .groupby(id_col)\
        .head(1)\
        .set_index(id_col)

    df_tmp_3 = df_tmp\
        .groupby(id_col)\
        .agg({
            id_col: 'count',
            'count': 'sum',
            agg_col: 'sum'
        })\
        .rename(columns={
            id_col: 'nunique', 
            'count': 'total_count',
            agg_col: 'total_sum'
        })

    df_res = pd.concat([df_tmp_2, df_tmp_3], axis=1)
    df_res = df_res.assign(
        count_ratio = df_res['count'] / df_res['total_count'],
        sum_ratio = df_res[agg_col] / df_res['total_sum']
    )\
    .rename(columns={
         column: f'most_comm_{column}',
        'nunique': f'nunique_{column}',
        'count_ratio': f'most_comm_cnt_ratio_{column}',
        'sum_ratio': f'most_comm_sum_ratio_{column}',
    }).drop(['count', agg_col, 'total_count', 'total_sum'], axis=1)
    
    del df_tmp, df_tmp_2, df_tmp_3
    gc.collect()
    
    return df_res


def repeated_cross_validate(
    df_full,
    upift_model,
    try_feat=None,
    pipeline_flg=False,
    target_col='target',
    treatment_col='treatment_flg',
    n_iter=10,
    test_size=0.3,
    seed=42,
    verbose=True,
    **metrics
):
    '''
    Make n_iter train train / val random data splits
    Fit model on train, score and calculate scores and metric on val set
    
    # Use _score function from sklearn.model_selection._validation
    # Add return estimator param like in cross_validate
    '''
    
    out_res = {}
    train_scores = []
    test_scores = []
    fitted_models = []
    
    if try_feat is None:
        try_feat = df_full.columns.tolist()
    
    for iter_ in range(1, (n_iter+1)):
        
        if verbose:
            print(f'Iteration number: {iter_}')
        
        df_tr, df_val = train_test_split(
            df_full, 
            test_size=test_size,
            random_state=seed+iter_
        )
        
        if pipeline_flg:
            upift_model.fit(
                X=df_tr[try_feat],
                y=df_tr[target_col],
                est__treatment=df_tr[treatment_col]
            )
            
        else:
            upift_model.fit(
                X=df_tr[try_feat],
                y=df_tr[target_col],
                treatment=df_tr[treatment_col]
            )
        
        pred_tr  = upift_model.predict(df_tr[try_feat])
        pred_val = upift_model.predict(df_val[try_feat])
        
        # get metrics value:
        metrics_train = {}
        metrics_val = {}
        for metric_name, metrics_func in metrics.items():
            
            score_tr = metrics_func(
              y_true=df_tr[target_col].values, 
              uplift=pred_tr, 
              treatment=df_tr[treatment_col].values
            )
            
            score_val = metrics_func(
                y_true=df_val[target_col].values, 
                uplift=pred_val, 
                treatment=df_val[treatment_col].values,
            )
            
            metrics_train[metric_name] = score_tr
            metrics_val[metric_name] = score_val
        
        train_scores.append(metrics_train)
        test_scores.append(metrics_val)
        fitted_models.append(upift_model)
       
    out_res = {
        'train_scores': train_scores,
        'test_scores': test_scores,
        'fitted_models': fitted_models
    }
        
    return out_res


# old version of reapeted cross validation:
def make_validation(
    df_full,
    upift_model,
    try_feat,
    pipeline_flg=False,
    target_col='target',
    treatment_col='treatment_flg',
    n_iter=10,
    test_size=0.3,
    seed=42,
    verbose=True
):
    '''
    Make n_iter train train / val random data splits
    Fit model on train, score and calculate scores and metric on val set
  
    '''
    out_res = {}
    for iter_ in range(1, (n_iter+1)):
        
        if verbose:
            print(f'Iteration number: {iter_}')
        
        df_tr, df_val = train_test_split(
            df_full, 
            test_size=test_size,
            random_state=(seed + iter_)
        )
        
        if pipeline_flg:
            upift_model.fit(
                X=df_tr[try_feat],
                y=df_tr[target_col],
                est__treatment=df_tr[treatment_col]
            )
            
        else:
            upift_model.fit(
                X=df_tr[try_feat],
                y=df_tr[target_col],
                treatment=df_tr[treatment_col]
            )
        
        pred_tr  = upift_model.predict(df_tr[try_feat])
        pred_val = upift_model.predict(df_val[try_feat])
        
        score_tr = np.round(uplift_at_k(
            y_true=df_tr[target_col].values, 
            uplift=pred_tr, 
            treatment=df_tr[treatment_col].values, 
            k=0.3
        ),5)
        
        score_val = np.round(uplift_at_k(
            y_true=df_val[target_col].values, 
            uplift=pred_val, 
            treatment=df_val[treatment_col].values, 
            k=0.3
        ),5)
        
        out_res[iter_] = {
            'score_tr': score_tr,
            'score_val': score_val
        }
        
    return out_res

