import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from eumap.misc import find_files, ttprint, nan_percentile, GoogleSheet
from eumap.raster import read_rasters, save_rasters
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

def read_features(file_path):
    with open(file_path, 'r') as file:
        features = [line.strip() for line in file.readlines()]
    return features

def find_knee(df):
    slopes = (df['accum'].diff(-1)) / (df['freq'].diff(-1))*(-1)
    knee_index = slopes.idxmax()
    return knee_index

def run_rankcv(data, covs, tgt, spatial_cv_column, weights_feature=None, n_bootstrap=20, ntrees = 100):

    groups = data[spatial_cv_column].unique()
    runs = []
    
    # loop on different bootstrap
    for k in range(n_bootstrap):
        
        np.random.seed(k)
        selected_groups = np.random.choice(groups, int(len(groups) * 0.7), False)  # each time cover 70% of the tiles
        samples_train = data[data[spatial_cv_column].isin(selected_groups)]
        train = samples_train.groupby(spatial_cv_column, group_keys=False).apply(lambda x: x.sample(min(len(x), 20)))  # make sure to select enough data for training
        
        ttprint(f'{k} iteration, training size: {len(train)}')
        # Get weights if applicable
        if weights_feature:
            weights = train[weights_feature].to_numpy()
            rf = RandomForestRegressor(random_state=41, n_jobs=80, n_estimators=ntrees)
            rf.fit(train[covs], train[tgt], sample_weight=weights)
        else:
            rf = RandomForestRegressor(random_state=41, n_jobs=80, n_estimators=ntrees)
            rf.fit(train[covs], train[tgt])

        importances = pd.Series(rf.feature_importances_, index=covs).sort_values(ascending=True)
        importances = importances[importances>=importances.mean()]
        runs.append((importances.index, np.array(importances.to_list())))
        
    result_rankcv = pd.DataFrame(
        dict(feature=[feature for run in runs for feature in run[0]], 
             importance=[importance for run in runs for importance in run[1]])
    )
    
    feature_list = result_rankcv.groupby(['feature']).count().rename(columns=dict(importance='freq')).reset_index()
    features_freq = feature_list.groupby('freq').count().reset_index().sort_values(by='freq', ascending=False)
    # features_freq['accum'] = features_freq['feature'].cumsum()
    # knee_index = find_knee(features_freq)
    # knee_freq = features_freq.loc[knee_index]['freq']
    # minf = features_freq.loc[features_freq['feature']==features_freq['feature'].min(),'freq'].values[0]
    minv = 15
    covs = feature_list[feature_list['freq']>=minv]['feature'].tolist() # choose only those with high frequency
    feature_list.to_csv(f'{output_folder}/benchmark_rank.freq.covs_{prop}.csv',index=False)
    while len(covs)>90 and minv<20:
        minv = minv+1
        covs = feature_list[feature_list['freq']>=minv]['feature'].tolist()
    if 'hzn_dep' not in covs:
        covs.append('hzn_dep')

    print(f'--------------{len(covs)} features selected for {prop}, threshold: {minv}---------')

    # Save the final list of features for records
    with open(f'{output_folder}/benchmark_selected.covs_{prop}.txt', 'w') as file:
        for item in covs:
            file.write(f"{item}\n")
            
    return covs

def calc_ccc(y_true, y_pred):
    if len(y_true) <= 1 or len(y_pred) <= 1:
        return np.nan  # Return NaN if there is insufficient data

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    
    try:
        cov_matrix = np.cov(y_true, y_pred)
        covariance = cov_matrix[0, 1]
        var_true = cov_matrix[0, 0]
        var_pred = cov_matrix[1, 1]
    except Warning:
        warnings.warn("Covariance calculation encountered an issue.")
        return np.nan  # Return NaN if covariance calculation fails

    if var_true + var_pred + (mean_true - mean_pred) ** 2 == 0:
        return np.nan  # Avoid division by zero

    ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
    return ccc

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom <= 0 for slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in multiply")


def accuracy_plot(y_test, y_pred, space, title, output_folder, show_range=[0, 7], vmax=20):
    mae, mape, ccc, r2 = calc_metrics(y_test, y_pred, space)
    
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(10, 8))  # Slightly larger figure to accommodate the colorbar
    fig.suptitle(title, fontsize=20, fontweight='bold')
    ax.set_title(f'R2={r2:.2f}, MAE={mae:.2f}, CCC={ccc:.2f}')
    hb = ax.hexbin(y_test, y_pred, gridsize=(150, 150), cmap='plasma_r', mincnt=1, vmax=vmax)
    ax.set_xlabel('True')
    ax.set_ylabel('Pred')
    ax.set_aspect('auto', adjustable='box')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.plot(show_range, show_range, "-k", alpha=.5)
    
    # Create a colorbar with proper spacing
    cax = fig.add_axes([ax.get_position().x1 + 0.05, ax.get_position().y0, 0.02, ax.get_position().height])
    cb = fig.colorbar(hb, cax=cax)
    
    plt.tight_layout(rect=[0, 0, 0.92, 1])  # Adjust the right margin to make room for colorbar
    
    plt.savefig(f'{output_folder}/plot_accuracy_{title}.pdf', format='pdf', bbox_inches='tight', dpi=300)
    return r2, mae, ccc

def error_spatial_plot(y_test, y_pred, lat, lon, title, output_folder, latbox=[33, 72], lonbox=[-12, 35]):
    y_error = y_pred - y_test
    fig, ax = plt.subplots(figsize=(11, 8))

    hexbin = ax.hexbin(lon, lat, C=y_error, gridsize=100, cmap='seismic', mincnt=1,
                       reduce_C_function=np.mean)

    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)
    ax.set_title(f'{title} - error', fontsize=16)

    # Set axis limits if provided
    if latbox is not None:
        ax.set_ylim(latbox)
    if lonbox is not None:
        ax.set_xlim(lonbox)

    # Create colorbar with proper alignment
    colorbar = fig.colorbar(hexbin, ax=ax, pad=0.01)
    colorbar.set_label('Prediction Error', fontsize=14)

    plt.grid(True)

    # Use tight_layout with custom rect to avoid cutting off labels or the colorbar
    plt.tight_layout(rect=[0, 0, 0.92, 1])  # Adjust these values as necessary

    plt.savefig(f'{output_folder}/plot_error_spatial_{title}.pdf', format='pdf', dpi=300, bbox_inches='tight')


# use sorted plot to check extrapolation problem
def sorted_plot(y_test, y_pred, title, output_folder):
    # Sort values for a cleaner plot
    sorted_indices = np.argsort(y_test) # sort according to true y values, get the sorted index
    sorted_y_test = np.array(y_test)[sorted_indices] # sort with the index
    sorted_y_pred = y_pred[sorted_indices]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(sorted_y_test)), sorted_y_pred, 'or', label='Predicted Values')
    plt.plot(range(len(sorted_y_test)), sorted_y_test, 'k-', label='True Values', alpha=0.4)

    # Formatting
    plt.title(f'check end values of {title}')
    plt.xlabel('Data Points (sorted by true values)')
    plt.ylabel('Predicted/True Values')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_folder}/plot_sorted_{title}.pdf', format='pdf', dpi=300)
    
def calc_metrics(y_true, y_pred, space):
    if space == 'normal':
        mae = mean_absolute_error(y_true, y_pred)
        medae = median_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        bias = np.nanmean(y_pred-y_true)
        ccc = calc_ccc(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
    else:
        ccc = calc_ccc(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # report MAE and MAPE in original scale
        y_true = np.expm1(y_true)
        y_pred = np.expm1(y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        medae = median_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        bias = np.nanmean(y_pred-y_true)
    return mae, medae, mape, ccc, r2, bias


def run_cumusort(data, tgt, prop, space, output_folder, weights_feature=None, threshold_step=0.001):
    covs_all = read_features(f'/home/opengeohub/xuemeng/work_xuemeng/soc/SOC-EU/features/001_covar_all.txt')
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
    
    max_threshold = sorted_importances['Mean Cumulative Feature Importance'].max()
    thresholds = np.arange(0, max_threshold + threshold_step, threshold_step)
    previous_feature_set = set([])
    n_splits = 5
    results = [] # store the results
    
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

        metrics = calc_metrics(y_true, y_pred, space)
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
        if row['Num_Features'] < 100:
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

    # Vertical line for the best and selected threshold
    ax1.axvline(x=best_threshold, color='grey', linestyle='--', label='Best Threshold')
    ax1.axvline(x=selected_threshold, color='cyan', linestyle='--', label='Selected Threshold')
    
    # Update the legend to include the vertical line label
    lines += [ax1.axvline(x=best_threshold, color='grey', linestyle='--')]
    labels += ['Best Threshold']
    lines += [ax1.axvline(x=selected_threshold, color='cyan', linestyle='--')]
    labels += ['Selected Threshold']
    ax1.legend(lines, labels, loc='upper right', fontsize=14, framealpha=0.5)#, bbox_to_anchor=(0.15, 0.95)

    plt.title(f'{prop}, {best_num_features} features', fontsize=16)
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

#     # simple ANN
#     ttprint('----------------------ann------------------------')
#     warnings.filterwarnings('ignore')
#     pipeline = Pipeline([
#         ('scaler', StandardScaler()),
#         ('mlp', MLPRegressor(max_iter=5000, early_stopping=True, random_state=28))
#     ])
#     param_ann = {
#         'mlp__hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100)],  # NN structure
#         'mlp__activation': ['tanh', 'relu', 'logistic'],  # commonly used activation functions in NN
#         'mlp__solver': ['sgd','adam'],  # optimizer set as sgd
#         'mlp__alpha': [0.0001, 0.001, 0.01],  # regularization to prevent overfitting
#         'mlp__learning_rate': ['constant', 'adaptive'],  # how aggressive the weights update
#         'mlp__learning_rate_init': [0.001, 0.01]  # initial learning rate

#     }
#     tune_ann = HalvingGridSearchCV(
#         estimator=pipeline,
#         param_grid=param_ann,
#         scoring=fitting_score,
#         n_jobs=90,
#         cv=cv,
#         verbose=1,
#         random_state=1993
#     )
#     tune_ann.fit(cal[covs], cal[tgt], groups=cal[spatial_cv_column])
#     ann = tune_ann.best_estimator_
#     joblib.dump(ann, f'{output_folder}/model_ann.{prop}_ccc.joblib')
#     models.append(ann)
#     model_names.append('ann')

#     # # lightGBR
#     import lightgbm as lgb
#     ttprint('----------------------lightGBM------------------------')
#     def clean_feature_names(df):
#         df.columns = [col.replace('{', '').replace('}', '').replace(':', '').replace(',', '').replace('"', '') for col in df.columns]
#         return df
#     from sklearn.preprocessing import FunctionTransformer
#     clean_names_transformer = FunctionTransformer(clean_feature_names, validate=False)
#     pipeline = Pipeline([
#         ('clean_names', clean_names_transformer),  # Clean feature names
#         ('lgbm', lgb.LGBMRegressor(random_state=35,verbose=-1))         # Replace with any model you intend to use
#     ])
#     param_lgb = {
#         'lgbm__n_estimators': [80, 100, 120],  # Lower initial values for quicker testing
#         'lgbm__max_depth': [3, 5, 7],  # Lower maximum depths
#         'lgbm__num_leaves': [20, 31, 40],  # Significantly fewer leaves
#         'lgbm__learning_rate': [0.01, 0.05, 0.1],  # Fine as is, covers a good range
#         'lgbm__min_child_samples': [20, 30, 50],  # Much lower values to accommodate small data sets
#         'lgbm__subsample': [0.8, 1.0],  # Reduced range, focusing on higher subsampling
#         'lgbm__colsample_bytree': [0.8, 1.0],  # Less variation, focus on higher values
#         'lgbm__verbosity': [-1]
#     }
#     tune_lgb = HalvingGridSearchCV(
#         estimator=pipeline,
#         param_grid=param_lgb,
#         scoring=fitting_score,
#         n_jobs=90,
#         cv=cv,
#         verbose=1,
#         random_state=1994
#     )
#     tune_lgb.fit(cal[covs], cal[tgt], groups=cal[spatial_cv_column])
#     lgbmd = tune_lgb.best_estimator_
#     joblib.dump(lgbmd, f'{output_folder}/model_lgb.{prop}_ccc.joblib')
#     models.append(lgbmd)
#     model_names.append('lgb')
    
    # ## weighted version
    sample_weights = cal[f'{prop}_qa'].values #**2
    ttprint('----------------------weighted rf------------------------')
    # random forest
    tune_rf.fit(cal[covs], cal[tgt], sample_weight=sample_weights, groups=cal[spatial_cv_column])
    rf_weighted = tune_rf.best_estimator_
    joblib.dump(rf_weighted, f'{output_folder}/model_rf.{prop}_ccc.weighted.joblib')
    models.append(rf_weighted)
    model_names.append('rf_weighted')
#     # lightGBM
#     ttprint('----------------------weighted lightGBM------------------------')
#     fit_params = {'lgbm__sample_weight': sample_weights}
#     tune_lgb.fit(cal[covs], cal[tgt], **fit_params, groups=cal[spatial_cv_column])
#     lgb_weighted = tune_lgb.best_estimator_
#     joblib.dump(lgb_weighted, f'{output_folder}/model_lgb.{prop}_ccc.weighted.joblib')
#     models.append(lgb_weighted)
#     model_names.append('lgb_weighted')
    
    return models, model_names
    
    
def evaluate_model(models,model_names,train,covs,prop,space,output_folder,test=None):
    # set target variable
    if space=='log1p':
        tgt = f'{prop}_log1p'
    else:
        tgt = prop
        
    # cv, test
    train = train.dropna(subset=covs,how='any')
    
    sample_weights = train[f'{prop}_qa'].values**2
    results = []
    show_low = math.floor(train[tgt].min())
    show_high = math.ceil(train[tgt].max())
    cv = GroupKFold(n_splits=5) 
    for im in range(len(models)):
        model_name = model_names[im]
        model = models[im]
        figure_name = model_name
        ttprint(figure_name)
        fit_params = {}
        # Determine the last step name early if it's a pipeline
        if hasattr(model, 'named_steps'):
            last_step_name = list(model.named_steps.keys())[-1]
            if 'weighted' in model_name:
                fit_params = {f'{last_step_name}__sample_weight': sample_weights}
        elif 'weighted' in model_name:
            fit_params = {'sample_weight': sample_weights}
            
        
        start_time = time.time()
        y_pred_cv = cross_val_predict(model, train[covs], train[tgt], cv=cv, groups=train['tile_id'], n_jobs=90, fit_params=fit_params)
        end_time = time.time()
        cv_time = (end_time - start_time)
        r2_cv, mae_cv, ccc_cv = accuracy_plot(train[tgt], y_pred_cv, space, figure_name+ '-cv', output_folder=output_folder, show_range = [show_low, show_high], vmax=20) 
        
        if test is not None:
            test = test.dropna(subset=covs,how='any')
            start_time = time.time()
            model.fit(train[covs], train[tgt], **fit_params)
            y_pred_val = model.predict(test[covs])
            end_time = time.time()
            testing_time = (end_time - start_time)
            r2_val, mae_val, ccc_val = accuracy_plot(test[tgt], y_pred_val, figure_name+ '-test', output_folder=output_folder,show_range = [show_low, show_high], vmax=5) 
            error_spatial_plot(test[tgt], y_pred_val, test['lat'], test['lon'], figure_name+ '-test', output_folder=output_folder)
            sorted_plot(test[tgt],y_pred_val, figure_name+ '-test', output_folder=output_folder)

        # store the metrics
        results.append({
            'title': model_name,
            'R2_cv': r2_cv,
            'MAE_cv': mae_cv,
            'CCC_cv': ccc_cv,
            # 'R2_val': r2_val,
            # 'MAE_val': mae_val,
            # 'CCC_val': ccc_val,
            'cv_time (s)': cv_time,
            # 'test_time (s)': testing_time
        })
        
#         # store feature importance
#         if hasattr(model, 'named_steps'):  # Check if it's a pipeline
#             last_step = model.named_steps[last_step_name]
#             if hasattr(last_step, 'feature_importances_'):
#                 importances = last_step.feature_importances_
#         elif hasattr(model, 'feature_importances_'):  # Direct model
#             importances = model.feature_importances_
#         else:
#             importances = [0] * len(covs)  # Default to zero if no importances are available

#         feature_importance_df = pd.DataFrame({
#             'feature': covs,
#             'importance': importances
#         })
#         sorted_feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
        # sorted_feature_importance_df.to_csv(f'{output_folder}/feature.importances_{prop}_{model_name}.txt', index=False, sep='\t')
          
    results = pd.DataFrame(results)
    results.to_csv(f'{output_folder}/benchmark_metrics_{prop}.csv',index=False)
    return results



def separate_data(prop, filt, space, output_folder, df): 
    # df = pd.read_csv(f'/home/opengeohub/xuemeng/work_xuemeng/soc/data/002_data_whole.csv',low_memory=False) 
    os.makedirs(output_folder, exist_ok=True)
    
    ### data set preparation
    # clean the data according to each properties
    df = df.loc[df[prop].notna()]
    df = df.loc[df[f'{prop}_qa']>filt]
    # df[prop].hist(bins=40)
    
    # set target variable
    if space=='log1p':
        df.loc[:,f'{prop}_log1p'] = np.log1p(df[prop])
        tgt = f'{prop}_log1p'
    else:
        tgt = prop
            
    # split calibration, train and test for benchmark
    bd_val = pd.read_csv(f'/home/opengeohub/xuemeng/work_xuemeng/soc/data/003.0_validate.pnts.rob_bd.csv',low_memory=False)
    oc_val = pd.read_csv(f'/home/opengeohub/xuemeng/work_xuemeng/soc/data/003.1_validate.pnts.rob_soc.csv',low_memory=False)
    idl = bd_val['id'].values.tolist() + oc_val['id'].values.tolist()
    idl = [str(i) for i in idl]
    test = df.loc[df['id'].isin(idl)] # individual test datasets
    cal_train = df.loc[~df['id'].isin(idl)] # calibration and train
    # get 10% of training data as calibration for parameter fine tuning and feature selection
    cal_train.reset_index(drop=True, inplace=True)
    cal = cal_train.groupby('tile_id', group_keys=False).apply(lambda x: x.sample(n=max(1, int(np.ceil(0.16 * len(x))))))
    # the rest as training dataset
    train = cal_train.drop(cal.index)
    # if test_size>0:
    #     test = test.iloc[0:round(len(test)*test_size)]
    #     train = train.iloc[0:round(len(train)*test_size)]
    #     cal = cal.iloc[0:round(len(cal)*test_size)]
    print(f'calibration size {len(cal)}, training size {len(train)}, test size {len(test)}')
    print(f'sum {len(cal)+len(train)+len(test)}, df {len(df)}')
    print(f'ratio cal:trai - {len(cal)/len(train):.2f}')
    cal.to_csv(f'{output_folder}/benchmark_cal.pnts_{prop}.csv',index=False)
    train.to_csv(f'{output_folder}/benchmark_train.pnts_{prop}.csv',index=False)
    test.to_csv(f'{output_folder}/benchmark_test.pnts_{prop}.csv',index=False)
    return cal, train, test



def run_benchmark(folder,output_folder,space,prop,filt,test_size=0):
    cal, train, test = separate_data(prop, filt, space, output_folder)
    
    # # read in data
    # cal.to_csv(f'{output_folder}/benchmark_cal.pnts_{prop}.csv',index=False)
    # train.to_csv(f'{output_folder}/benchmark_train.pnts_{prop}.csv',index=False)
    # test.to_csv(f'{output_folder}/benchmark_test.pnts_{prop}.csv',index=False)
    
    cal = pd.read_csv(f'{output_folder}/benchmark_cal.pnts_{prop}.csv',low_memory=False)
    covs = run_cumusort(cal, tgt, prop, output_folder) #, weights_feature=None, threshold_step=0.001
    models, model_names = parameter_fine_tuning(cal, covs, tgt, output_folder)
    
    
    train = pd.read_csv(f'{output_folder}/benchmark_train.pnts_{prop}.csv',low_memory=False)
    test = pd.read_csv(f'{output_folder}/benchmark_test.pnts_{prop}.csv',low_memory=False)
    evaluate_model(models,model_name,train,test,covs,tgt,prop)
    return None
    


def calc_picp(lower_bounds, upper_bounds, true_values):
    within_bounds = np.sum((true_values >= lower_bounds) & (true_values <= upper_bounds))
    picp = within_bounds / len(true_values)
    return picp

def calc_qcp(predictions, true_values, quantile):
    return np.mean(true_values <= predictions)

def quantile_cross_val_predict(estimator, X, y, cv, quantiles, groups=None):
    """ Custom cross-validation to handle quantile predictions with group support. """
    predictions = {q: [] for q in quantiles}
    if groups is None:
        cv_split = cv.split(X)
    else:
        cv_split = cv.split(X, y, groups)
    
    for train_idx, test_idx in cv_split:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        
        # Fit the model on the training data
        estimator.fit(X_train, y_train)
        
        # Predict each quantile and store results
        for q in quantiles:
            pred = estimator.predict(X_test, quantiles=q)
            predictions[q].extend(pred)
    
    return predictions



    
# # put all of them on s3
# from minio import Minio
# def mover(file):
#     s3_config = {
#         'access_key': 'iwum9G1fEQ920lYV4ol9',
#         'secret_access_key': 'GMBME3Wsm8S7mBXw3U4CNWurkzWMqGZ0n2rXHggS0',
#         'host': '192.168.1.30:8333',
#         'bucket': 'ai4sh-landmasked'}
#     client = Minio(s3_config['host'], s3_config['access_key'], s3_config['secret_access_key'], secure=False)
#     # .joinpath('p50').joinpath(str(tile)).joinpath(f'{da}.tif'
#     s3_path = f"model_benchmark/{prop}/{file.split('/')[-1]}"
#     client.fput_object(s3_config['bucket'], s3_path, file)
#     # os.remove(file)
#     return None

# result_content = find_files(f'{output_folder}/',f'*{prop}*')
# for i in result_content:
#     mover(i)


# # cubist
# pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('cubist', Cubist())
# ])
# param_cubist = {
#     'cubist__n_rules': [100, 300, 500],  # number of rules to be generated
#     'cubist__n_committees': [1, 5, 10],  # committee: ensembles of models
#     'cubist__neighbors': [None, 3, 6, 9],  # number of nearest neighbors to use when making a prediction
#     'cubist__unbiased': [False, True],  # whether or not to use an unbiased method of rule generation
#     'cubist__extrapolation': [0.02, 0.05],  # limits the extent to which predictions can extrapolate beyond the range of the calibration data, a fraction of the total range of the target variable
#     'cubist__sample': [None]  # fraction of the calibration data used in building each model, since the calibration dataset could be very small
# }
# tune_cubist = HalvingGridSearchCV(
#     estimator=pipeline,
#     param_grid=param_cubist,
#     scoring=fitting_score,
#     n_jobs=90,
#     cv=cv
# )
# X_cal = pd.DataFrame(cal[covs].values, columns=covs)
# y_cal = cal[tgt]
# tune_cubist.fit(X_cal, y_cal, groups=cal[spatial_cv_column])
# cubist = tune_cubist.best_estimator_
# joblib.dump(cubist, f'{output_folder}/model_cubist.{prop}_{space}.ccc.joblib')


# # cubist
# fit_params = {'cubist__sample_weight': sample_weights}
# tune_cubist.fit(X_cal, y_cal, **fit_params, groups=cal[spatial_cv_column])
# cubist_weighted = tune_cubist.best_estimator_
# joblib.dump(cubist_weighted, f'{output_folder}/model_cubist.{prop}_{space}.ccc.weighted.joblib')