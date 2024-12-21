import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from skmap.overlay import SpaceOverlay, SpaceTimeOverlay
from skmap.misc import find_files, GoogleSheet, ttprint
import warnings
import multiprocess as mp
import time
from scipy.special import expit, logit
import warnings
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, cross_val_score, HalvingGridSearchCV, KFold, GroupKFold
from sklearn.model_selection import RandomizedSearchCV, HalvingRandomSearchCV, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, median_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
import joblib
import pickle
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
# from cubist import Cubist
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path
import os
from scipy.stats import randint, uniform
import math

def weighted_mean(values, weights=None):
    if weights is None:
        return np.mean(values)
    return np.sum(values * weights) / np.sum(weights)

def weighted_mae(y_true, y_pred, weights=None):
    if weights is None:
        return np.mean(np.abs(y_true - y_pred))
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)

def weighted_mape(y_true, y_pred, weights=None):
    return np.mean(np.abs((y_true - y_pred) / y_true))

def weighted_bias(y_true, y_pred, weights=None):
    if weights is None:
        return np.mean(y_pred - y_true)
    return np.sum(weights * (y_pred - y_true)) / np.sum(weights)

def weighted_medae(y_true, y_pred, weights=None):
    return np.median(np.abs(y_true - y_pred))

def weighted_ccc(y_true, y_pred, weights=None):
    if weights is None:
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
        var_true = np.mean((y_true - mean_true) ** 2)
        var_pred = np.mean((y_pred - mean_pred) ** 2)
    else:
        mean_true = weighted_mean(y_true, weights)
        mean_pred = weighted_mean(y_pred, weights)
        cov = np.sum(weights * (y_true - mean_true) * (y_pred - mean_pred)) / np.sum(weights)
        var_true = np.sum(weights * (y_true - mean_true) ** 2) / np.sum(weights)
        var_pred = np.sum(weights * (y_pred - mean_pred) ** 2) / np.sum(weights)
    
    ccc = (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
    return ccc

def calc_metrics(y_true, y_pred, weights=None, space='normal'):
    if space == 'normal':
        mae = weighted_mae(y_true, y_pred, weights)
        medae = weighted_medae(y_true, y_pred, weights)
        mape = weighted_mape(y_true, y_pred, weights)
        bias = weighted_bias(y_true, y_pred, weights)
        ccc = weighted_ccc(y_true, y_pred, weights)
        r2 = r2_score(y_true, y_pred, sample_weight=weights)
    else:
        ccc = weighted_ccc(y_true, y_pred, weights)
        r2 = r2_score(y_true, y_pred, sample_weight=weights)
        
        y_true = np.expm1(y_true)
        y_pred = np.expm1(y_pred)
        mae = weighted_mae(y_true, y_pred, weights)
        medae = weighted_medae(y_true, y_pred, weights)
        mape = weighted_mape(y_true, y_pred, weights)
        bias = weighted_bias(y_true, y_pred, weights)
        
    return mae, medae, mape, ccc, r2, bias



def read_features(file_path):
    with open(file_path, 'r') as file:
        features = [line.strip() for line in file.readlines()]
    return features

def find_knee(df):
    slopes = (df['accum'].diff(-1)) / (df['freq'].diff(-1))*(-1)
    knee_index = slopes.idxmax()
    return knee_index


# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom <= 0 for slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in multiply")



def cfi_calc(data, tgt, prop, space, output_folder, covs_all, weights_feature=None):
    data = data.dropna(subset=covs_all,how='any')
    n_bootstrap=20
    ntrees = 100
    
    spatial_cv_column='tile_id'
    groups = data[spatial_cv_column].unique()
    runs = []
    feature_importances = []
    
    print(f'start bootstrap on different subset...')
    for k in range(n_bootstrap):
        
        np.random.seed(k)
        selected_groups = np.random.choice(groups, int(len(groups) * 0.7), False)  # each time cover 70% of the tiles
        train = data[data[spatial_cv_column].isin(selected_groups)]
        # train = train.groupby(spatial_cv_column, group_keys=False).apply(lambda x: x.sample(min(len(x), 5)))  # make sure to select enough data for training
        
        ttprint(f'{k} iteration, training size: {len(train)}')
        # Get weights if applicable
        if weights_feature:
            weights = train[weights_feature].to_numpy()
            rf = RandomForestRegressor(random_state=41, n_jobs=80, n_estimators=ntrees)
            rf.fit(train[covs_all], train[tgt], sample_weight=weights)
        else:
            rf = RandomForestRegressor(random_state=41, n_jobs=80, n_estimators=ntrees)
            rf.fit(train[covs_all], train[tgt])
            
        feature_importances.append(rf.feature_importances_)
        
    result = pd.DataFrame(feature_importances, columns=covs_all)
        
    
    sorted_importances = result.mean(axis=0).sort_values(ascending=False)
    sorted_importances = sorted_importances.reset_index()
    sorted_importances.columns = ['Feature Name', 'Mean Cumulative Feature Importance']
    sorted_importances.to_csv(f'{output_folder}/cumulative.feature.importance_{prop}.csv',index=False)
    
    return sorted_importances
    
def rscfi(data, tgt, prop, space, output_folder, covs_all, sorted_importances, threshold_step=0.001, threshold_number = 100):
    max_threshold = sorted_importances['Mean Cumulative Feature Importance'].max()
    thresholds = np.arange(0, max_threshold + threshold_step, threshold_step)
    previous_feature_set = set([])
    results = []
    data = data.dropna(subset=covs_all,how='any')
    
    n_splits = 5
    ntrees = 100
    spatial_cv_column='tile_id'
    groups = data[spatial_cv_column].unique()
    
    print(f'start feature elimination evaluation...')
    for threshold in thresholds:
        current_features_df = sorted_importances[sorted_importances['Mean Cumulative Feature Importance'] >= threshold]
        current_features = current_features_df['Feature Name'].tolist()
        
        if set(current_features) == previous_feature_set:
            continue  # Skip if feature set doesn't change
        previous_feature_set = set(current_features)

        if len(current_features)<2:
            break  # Stop if limited (<2) features are left

        ttprint(f'processing {threshold} ...')
        rf = RandomForestRegressor(random_state=41, n_jobs=80, n_estimators=ntrees)
        group_kfold = GroupKFold(n_splits=n_splits)

        groups = data[spatial_cv_column].values
        y_pred = cross_val_predict(rf, data[current_features], data[tgt], cv=group_kfold, groups=groups, n_jobs=-1)
        y_true = data[tgt]

        metrics = calc_metrics(y_true, y_pred, weights=None, space=space)
        results.append((threshold, len(current_features), *metrics))
    
    results_df = pd.DataFrame(results, columns=['Threshold', 'Num_Features', 'MAE', 'MedAE', 'MAPE', 'CCC', 'R2','bias'])
    results_df['MAE_Rank'] = results_df['MAE'].rank(ascending=True)
    results_df['MAPE_Rank'] = results_df['MAPE'].rank(ascending=True)
    results_df['MedAE_Rank'] = results_df['MedAE'].rank(ascending=True)
    results_df['bias_Rank'] = results_df['bias'].rank(ascending=True)
    results_df['CCC_Rank'] = results_df['CCC'].rank(ascending=False)
    results_df['R2_Rank'] = results_df['R2'].rank(ascending=False)
    results_df['Combined_Rank'] = results_df['MAE_Rank'] + results_df['CCC_Rank'] + results_df['R2_Rank'] + results_df['MedAE_Rank']
    
    # select threshold
    results_df = results_df.sort_values(by='Combined_Rank')
    results_df.to_csv(f'{output_folder}/metrics.rank_feature.elimination_{prop}.csv', index=False)
    best_threshold = results_df.loc[results_df['Combined_Rank'].idxmin(), 'Threshold']

    for index, row in results_df.iterrows():
        if row['Num_Features'] < threshold_number:
            selected_threshold = row['Threshold']
            break
           
    features_df = sorted_importances[sorted_importances['Mean Cumulative Feature Importance'] >= selected_threshold]
    covs = features_df['Feature Name'].tolist()
    if 'hzn_dep' not in covs:
        print(f'{prop} model did not select depth as covs, adding it')
        covs.append('hzn_dep')
    print(f'--------------{len(covs)} features selected for {prop}, mean cumulative feature importance threshold: {selected_threshold}---------')
    with open(f'{output_folder}/benchmark_selected.covs_{prop}.txt', 'w') as file:
        for item in covs:
            file.write(f"{item}\n")
        
    results_df = results_df.sort_values(by='Threshold')
    # plot feature elimination analysis
    fig, ax1 = plt.subplots(figsize=(11, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Feature Importance Threshold', fontsize=16)
    ax1.set_ylabel('Number of Features', color=color, fontsize=16)
    line1 = ax1.plot(results_df['Threshold'], results_df['Num_Features'], color=color, marker='o', label='Num_Feat')
    ax1.tick_params(axis='y', labelcolor=color, labelsize=14)
    ax1.tick_params(axis='x', labelsize=14)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Evaluation Metrics', color=color, fontsize=16)
    line2 = ax2.plot(results_df['Threshold'], results_df['CCC'], color='tab:green', marker='x', linestyle='-', linewidth=2, label='CCC')
    line3 = ax2.plot(results_df['Threshold'], results_df['R2'], color='tab:red', marker='s', linestyle='-', linewidth=2, label='R2')
    line4 = ax2.plot(results_df['Threshold'], results_df['MAE']/10, color='tab:orange', marker='^', linestyle='-', linewidth=2, label='MAE (scaled by 0.1)')
    # line5 = ax2.plot(results_df['Threshold'], results_df['MAPE'], color='tab:purple', marker='d', linestyle='-', linewidth=2, label='MAPE')

    ax2.tick_params(axis='y', labelcolor=color, labelsize=14)

    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to avoid cutting off the title

    # Combine legends
    lines = line1 + line2 + line3 + line4 #+ line5
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(0.15, 0.95), fontsize=14, framealpha=0.5)

    # Best combined rank
    best_combined_rank_index = results_df['Combined_Rank'].idxmin()
    best_threshold = results_df.loc[best_combined_rank_index, 'Threshold']
    best_num_features = results_df.loc[best_combined_rank_index, 'Num_Features']
    selected_num_features = results_df.loc[results_df['Threshold'] == selected_threshold, 'Num_Features'].values[0]

    # Vertical line for the best and selected threshold
    ax1.axvline(x=best_threshold, color='grey', linestyle='--', label='Best Threshold')
    ax1.axvline(x=selected_threshold, color='cyan', linestyle='--', label='Selected Threshold')
    
    # Update the legend to include the vertical line label
    lines += [ax1.axvline(x=best_threshold, color='grey', linestyle='--')]
    labels += ['Best Threshold']
    lines += [ax1.axvline(x=selected_threshold, color='cyan', linestyle='--')]
    labels += ['Selected Threshold']
    ax1.legend(lines, labels, loc='upper right', fontsize=14, framealpha=0.5)#, bbox_to_anchor=(0.15, 0.95)

    plt.title(f'{prop}\nbest feature number: {best_num_features}, select {selected_num_features}', fontsize=16)
    plt.savefig(f'{output_folder}/plot_feature.elimination_{prop}.pdf')
    plt.show()
    
    return covs

    
def parameter_fine_tuning(cal, covs, tgt, prop, output_folder):
    models = [] #[rf, ann, lgb, rf_weighted, lgb_weighted] #cubist, cubist_weighted, 
    model_names = [] #['rf', 'ann', 'lgb', 'rf_weighted', 'lgb_weighted'] # 'cubist',, 'cubist_weighted'
    cal = cal.dropna(subset=covs,how='any')

    ### parameter fine tuning
    spatial_cv_column = 'tile_id'
    cv = GroupKFold(n_splits=5)
    ccc_scorer = make_scorer(calc_ccc, greater_is_better=True)
    fitting_score = ccc_scorer
    
    ## no weights version
    # random forest
    ttprint('----------------------rf------------------------')
    param_rf = {
        'n_estimators': [64],
        "criterion": [ 'squared_error', 'absolute_error', 'poisson', 'friedman_mse'],
        'max_depth': [10, 20, 30],
        'max_features': [0.3, 0.5, 0.7, 'log2', 'sqrt'],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    tune_rf = HalvingGridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid=param_rf,
        scoring=fitting_score,
        n_jobs=90, 
        cv=cv,
        verbose=1,
        random_state = 1992
    )
    tune_rf.fit(cal[covs], cal[tgt], groups=cal[spatial_cv_column])
    warnings.filterwarnings('ignore')
    rf = tune_rf.best_estimator_
    joblib.dump(rf, f'{output_folder}/model_rf.{prop}_ccc.joblib')
    models.append(rf)
    model_names.append('rf')
    
    # # # lightGBR
    # import lightgbm as lgb
    # ttprint('----------------------lightGBM------------------------')
    # def clean_feature_names(df):
    #     df.columns = [col.replace('{', '').replace('}', '').replace(':', '').replace(',', '').replace('"', '') for col in df.columns]
    #     return df
    # from sklearn.preprocessing import FunctionTransformer
    # clean_names_transformer = FunctionTransformer(clean_feature_names, validate=False)
    # pipeline = Pipeline([
    #     ('clean_names', clean_names_transformer),  # Clean feature names
    #     ('lgbm', lgb.LGBMRegressor(random_state=35,verbose=-1))         # Replace with any model you intend to use
    # ])
    # param_lgb = {
    #     'lgbm__n_estimators': [80, 100, 120],  # Lower initial values for quicker testing
    #     'lgbm__max_depth': [3, 5, 7],  # Lower maximum depths
    #     'lgbm__num_leaves': [20, 31, 40],  # Significantly fewer leaves
    #     'lgbm__learning_rate': [0.01, 0.05, 0.1],  # Fine as is, covers a good range
    #     'lgbm__min_child_samples': [20, 30, 50],  # Much lower values to accommodate small data sets
    #     'lgbm__subsample': [0.8, 1.0],  # Reduced range, focusing on higher subsampling
    #     'lgbm__colsample_bytree': [0.8, 1.0],  # Less variation, focus on higher values
    #     'lgbm__verbosity': [-1]
    # }
    # tune_lgb = HalvingGridSearchCV(
    #     estimator=pipeline,
    #     param_grid=param_lgb,
    #     scoring=fitting_score,
    #     n_jobs=90,
    #     cv=cv,
    #     verbose=1,
    #     random_state=1994
    # )
    # tune_lgb.fit(cal[covs], cal[tgt], groups=cal[spatial_cv_column])
    # lgbmd = tune_lgb.best_estimator_
    # joblib.dump(lgbmd, f'{output_folder}/model_lgb.{prop}_ccc.joblib')
    # models.append(lgbmd)
    # model_names.append('lgb')
    
    return models, model_names

def accuracy_plot(y_test, y_pred, prop, space, mdl, test_type, data_path):
    mae, medae, mape, ccc, r2, bias = calc_metrics(y_test, y_pred, weights=None, space=space)
    
    show_range = [
    math.floor(np.min([y_test.min(), y_pred.min()])),
    math.ceil(np.max([y_test.max(), y_pred.max()]))]
    vmax = 0.001*len(y_test)
    
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(10, 8))  # Slightly larger figure to accommodate the colorbar
    # fig.suptitle(title, fontsize=20, fontweight='bold', y=1.05)
    ax.set_title(f'{test_type}: {len(y_test)} points for {mdl} in {space} scale\nMAE={mae:.2f}, MedAE={medae:.2f}, bias = {bias:.2f},\nR2={r2:.2f},  CCC={ccc:.2f}')
    hb = ax.hexbin(y_pred, y_test, gridsize=(80, 80), cmap='plasma_r', mincnt=1, vmax=vmax)
    ax.set_xlabel(f'Predicted {prop}')
    ax.set_ylabel(f'Observed {prop}')
    ax.set_aspect('auto', adjustable='box')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.plot(show_range, show_range, "-k", alpha=.5)
    
    # Create a colorbar with proper spacing
    cax = fig.add_axes([ax.get_position().x1 + 0.05, ax.get_position().y0, 0.02, ax.get_position().height])
    cb = fig.colorbar(hb, cax=cax)
    
    plt.tight_layout(rect=[0, 0, 0.92, 1])  # Adjust the right margin to make room for colorbar
    
    plt.savefig(f'{data_path}/{prop}/plot_accuracy.{test_type}_{mdl}.{prop}.pdf', format='pdf', bbox_inches='tight', dpi=300)
    return mae, medae, mape, ccc, r2, bias


def evaluate_model(models,model_names,train,covs,prop,space,data_path,test=None,feature_importance=False):
    # set target variable
    if space=='log1p':
        tgt = f'{prop}_log1p'
    else:
        tgt = prop
        
    # cv, test
    train = train.dropna(subset=covs,how='any')
    results_cv = []
    results_val = []
    cv = GroupKFold(n_splits=5) 
    
    for im in range(len(models)):
        model_name = model_names[im]
        model = models[im]
        ttprint(f'accuracy evaluation for {model_name}')
        fit_params = {}
        
        # # Determine the last step name early if it's a pipeline
        # if hasattr(model, 'named_steps'):
        #     last_step_name = list(model.named_steps.keys())[-1]
        #     if 'weighted' in model_name:
        #         fit_params = {f'{last_step_name}__sample_weight': sample_weights}
        # elif 'weighted' in model_name:
        #     fit_params = {'sample_weight': sample_weights}
            
        start_time = time.time()
        y_pred_cv = cross_val_predict(model, train[covs], train[tgt], cv=cv, groups=train['tile_id'], n_jobs=90, fit_params=fit_params)
        end_time = time.time()
        cv_time = (end_time - start_time)
        mae_cv, medae_cv, mape_cv, ccc_cv, r2_cv, bias_cv = accuracy_plot(train[tgt], y_pred_cv, prop, space, model_name, 'spatial.cv', data_path) 
        
        # store the metrics
        results_cv.append({
            'title': model_name,
            'MAE_cv': mae_cv,
            'MedAE_cv': medae_cv,
            'MAPE_cv': mape_cv,
            'R2_cv': r2_cv,
            'CCC_cv': ccc_cv,
            'bias_cv': bias_cv,
            'cv_time (s)': cv_time
        })
        
        if test is not None:
            test = test.dropna(subset=covs,how='any')
            start_time = time.time()
            model.fit(train[covs], train[tgt])
            y_pred_val = model.predict(test[covs])
            end_time = time.time()
            testing_time = (end_time - start_time)
            mae_val, medae_val, mape_val, ccc_val, r2_val, bias_val = accuracy_plot(test[tgt], y_pred_val, prop, space, model_name, 'test', data_path) 
            # error_spatial_plot(test[tgt], y_pred_val, test['lat'], test['lon'], figure_name+ '-test', data_path=data_path)
            # sorted_plot(test[tgt],y_pred_val, figure_name+ '-test', data_path=data_path)
            
            # store the metrics
            results_val.append({
                'title': model_name,
                'MAE_val': mae_val,
                'MedAE_val': medae_val,
                'MAPE_val': mape_val,
                'R2_val': r2_val,
                'CCC_val': ccc_val,
                'bias_val': bias_val,
                'test_time (s)': testing_time
            })

            if feature_importance is not False:
                if feature_importance:
                    if hasattr(model, 'named_steps'):
                        last_step = model.named_steps[list(model.named_steps.keys())[-1]]
                        if hasattr(last_step, 'feature_importances_'):
                            impt = last_step.feature_importances_
                    elif hasattr(model, 'feature_importances_'):
                        impt = model.feature_importances_
                
                feature_importance_df = pd.DataFrame({
                    'feature': covs,
                    'importance': impt
                })
                sorted_feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
                sorted_feature_importance_df.to_csv(f'{data_path}/{prop}/feature.importances_{prop}_{model_name}.txt', index=False, sep='\t')
          
    results = pd.DataFrame(results_cv)
    if len(results_val)>0:
        results = pd.concat([results,pd.DataFrame(results_val)],axis=1)
    results.to_csv(f'{data_path}/{prop}/benchmark_metrics_{prop}.csv',index=False)
    return results

def separate_data(prop, space, output_folder, df): 
    # df = pd.read_csv(f'/home/opengeohub/xuemeng/work_xuemeng/soc/data/002_data_whole.csv',low_memory=False) 
    os.makedirs(output_folder, exist_ok=True)
    
    ### data set preparation
    # clean the data according to each properties
    df = df.loc[df[prop].notna()]
    # df[prop].hist(bins=40)
    
    # set target variable
    if space=='log1p':
        df.loc[:,f'{prop}_log1p'] = np.log1p(df[prop])
        tgt = f'{prop}_log1p'
    else:
        tgt = prop
       

    # Step 1: Identify single-sample classes in tile_id
    single_sample_classes = df['tile_id'].value_counts()[df['tile_id'].value_counts() == 1].index
    single_sample_df = df[df['tile_id'].isin(single_sample_classes)]
    multi_sample_df = df[~df['tile_id'].isin(single_sample_classes)]

    # Step 2: Randomly assign single-sample entries to cal, train, or test
    np.random.seed(42)  # For reproducibility
    single_sample_cal, single_sample_train, single_sample_test = np.split(
        single_sample_df.sample(frac=1, random_state=42), [len(single_sample_df) // 10, len(single_sample_df) * 2 // 10]
    )

    # Step 3: Perform the first stratified split on multi-sample classes
    # 80% of multi-sample data for training, 20% for temp (cal + test)
    temp, train = train_test_split(multi_sample_df, test_size=0.8, stratify=multi_sample_df['tile_id'], random_state=42)

    # Step 4: Identify classes in `temp` with 1 or 2 samples (small classes)
    small_classes_in_temp = temp['tile_id'].value_counts()[temp['tile_id'].value_counts() <= 2].index
    small_class_df = temp[temp['tile_id'].isin(small_classes_in_temp)]
    temp_remaining = temp[~temp['tile_id'].isin(small_classes_in_temp)]

    # Step 5: Perform second split on remaining data in `temp`
    # 10% each for cal and test
    cal, test = train_test_split(temp_remaining, test_size=0.5, stratify=temp_remaining['tile_id'], random_state=42)

    # Step 6: Randomly assign small classes to cal or test sets to preserve distribution without stratification issues
    small_cal, small_test = np.split(
        small_class_df.sample(frac=1, random_state=42), [len(small_class_df) // 2]
    )

    # Step 7: Combine all subsets
    cal = pd.concat([cal, single_sample_cal, small_cal])
    train = pd.concat([train, single_sample_train])
    test = pd.concat([test, single_sample_test, small_test])

    lsum = len(cal)+len(train)+len(test)
    print(f'size, calibration {len(cal)}, training {len(train)}, test {len(test)}')
    print(f'ratio, calibration {len(cal)/lsum:.2f}, training {len(train)/lsum:.2f}, test {len(test)/lsum:.2f}')
    print(f'sum {lsum}, df {len(df)}')
    
    cal.to_parquet(f'{output_folder}/data_cal_{prop}.pq')
    train.to_parquet(f'{output_folder}/data_train_{prop}.pq')
    test.to_parquet(f'{output_folder}/data_test_{prop}.pq')
    return cal, train, test


import matplotlib.pyplot as plt
def plot_top_features(prop, mdl, data_path, top_n=10):
    """
    Plots the top N features by importance in descending order.
    
    Parameters:
    - feature_importance_df (pd.DataFrame): A DataFrame with two columns: 
      'feature' and 'importance', sorted in descending order of importance.
    - top_n (int): Number of top features to plot. Default is 15.
    """
    feature_importance_df = pd.read_csv(f'{data_path}/{prop}/feature.importances_{prop}_{mdl}.txt', delimiter='\t') 
    # Ensure the DataFrame is sorted by importance in descending order
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    
    # Select the top N features
    top_features = feature_importance_df.head(top_n)
    
    # edit the feature names to make them less lengthy
    names = []
    for ii in top_features['feature'].to_list():
        if len(ii.split('_'))>5:
            kk = ii.split('_')[0] + '_' + ii.split('_')[1] + '_' + ii.split('_')[2] 
            if ('km_' in ii.split('_')[3]) & ('0m_' in ii.split('_')[3]):
                kk = kk + '_' + ii.split('_')[3]
        elif len(ii.split('_'))>2:
            kk = ii.split('_')[0] + '_' + ii.split('_')[1]
        else:
            kk = ii
            
        names.append(kk)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(names, top_features['importance'], color='skyblue')
    plt.xlabel('Feature importance')
    plt.title(f'{prop}\ntop {top_n} most important features')
    plt.gca().invert_yaxis()  # Invert y-axis to display the most important feature at the top
    # plt.show()
    
    plt.tight_layout()
    plt.savefig(f'{data_path}/{prop}/plot_feature.importance_{prop}.{mdl}.pdf')
    plt.show()  # Display the plot
    plt.close()
    
def plot_histogram(df, prop, space, data_path):
    if space == 'normal':
        plt.figure(figsize=(10, 6))
        plt.hist(df[prop], bins=40, alpha=0.75)
        plt.title(f'Histogram of {prop}\nin normal scale')
        # plt.xlabel(prop)
        plt.ylabel('Count')
        plt.savefig(f'{data_path}/{prop}/plot_histogram_{prop}.pdf', bbox_inches='tight')
        plt.show()
        plt.close()
    elif space == 'log1p':
        # Create the log1p transformed column
        df[prop + '_log1p'] = np.log1p(df[prop])

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

        # Original data histogram
        axes[0].hist(df[prop], bins=40, alpha=0.75)
        axes[0].set_title('In normal scale')
        # axes[0].set_xlabel(prop)

        # Transformed data histogram
        axes[1].hist(df[prop + '_log1p'], bins=40, alpha=0.75)
        axes[1].set_title('In log1p scale')
        # axes[1].set_xlabel(f'{prop}_log1p')

        # Setting a shared y-axis label
        fig.text(0.04, 0.5, 'Count', va='center', rotation='vertical', fontsize=18)
        fig.suptitle(f'Histograms of {prop}', fontsize=20)

        # Adjust layout and save to PDF
        plt.tight_layout(rect=[0.05, 0, 1, 1])  # Adjust layout to make room for the y-label
        plt.savefig(f'{data_path}/{prop}/plot_histogram_{prop}.pdf', format='pdf')
        plt.show()
        plt.close()
        
def pdp_hexbin(df, prop, space, mdl, data_path, fn = 3, grid_resolution=50, bins = None, cmap='viridis'):
    """
    Generate a partial dependence hexbin heatmap for a single feature.

    Parameters:
    - model: Trained model (should support `predict`).
    - df: DataFrame of input features.
    - feature: Feature name for which to generate PDP.
    - grid_resolution: Number of points to evaluate PDP along the feature's range.
    - bins: Binning method for hexbin ('log', 'linear', etc.).
    - cmap: Colormap for hexbin.
    - output_path: Path to save the plot. If None, the plot will be shown.
    """
    # get the top features that we would like to plot pdp for
    feature_importance_df = pd.read_csv(f'{data_path}/{prop}/feature.importances_{prop}_{mdl}.txt', delimiter='\t') 
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    top_features =  feature_importance_df.head(fn)['feature'].to_list()
    
    # load the model to do the pdp generation
    model = joblib.load(f'{data_path}/{prop}/model_rf.{prop}_ccc.joblib')
    
    # read in all the features
    covs = read_features(f'{data_path}/{prop}/benchmark_selected.covs_{prop}.txt')
    
    # prepare the data
    df = df.dropna(subset = covs+[prop])
    if space=='log1p':
        df.loc[:,f'{prop}_log1p'] = np.log1p(df[prop])
        tgt = f'{prop}_log1p'
    else:
        tgt = prop
        
    # start plotting
    idx = 0
    for feature in top_features:
        idx = idx+1
        feature_values = np.linspace(df[feature].min(), df[feature].max(), grid_resolution)
        pd_values = []

        # Iterate over the feature grid, calculate predictions while fixing the feature value
        for val in feature_values:
            # Copy the original data to avoid modifying it
            X_temp = df.copy()

            # Set the feature of interest to the current grid value for all samples
            X_temp[feature] = val

            # Predict and calculate the mean prediction for the fixed feature value
            mean_prediction = model.predict(X_temp[covs]).mean()
            pd_values.append(mean_prediction)
            
        if len(feature.split('_'))>5:
            fname = feature.split('_')[0] + '_' + feature.split('_')[1] + '_' + feature.split('_')[2] 
            if ('km_' in feature.split('_')[3]) & ('0m_' in feature.split('_')[3]):
                fname = fname + '_' + feature.split('_')[3]
        elif len(feature.split('_'))>2:
            fname = feature.split('_')[0] + '_' + feature.split('_')[1]
        else:
            fname = feature

        plt.figure(figsize=(8, 6))
        plt.hexbin(feature_values, pd_values, gridsize=30, mincnt=1, cmap=cmap, bins=bins if bins else None)
        plt.colorbar(label='Density')
        plt.xlabel(fname)
        plt.ylabel(prop)
        plt.title(f'{mdl} PDP')
        plt.savefig(f"{data_path}/{prop}/plot_pdp_top{idx}.{prop}.pdf", format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()