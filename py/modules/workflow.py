# -*- coding: utf-8 -*-

"""
 workflow.py
 Richard Wen (rwenite@gmail.com)

===============================================================

 A script for interfacing with input and output files via
 a workflow based approach. Handles progress saving by only
 incorporating file checks to see if a particular process
 has already been run, and skipping the processing step.

===============================================================
"""


"""
===============================================================
 Modules
===============================================================
"""


import base64
from io import BytesIO
from math import sqrt
from modules import helpers
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from statsmodels.robust.scale import mad
from treeinterpreter import treeinterpreter as ti


import geopandas as gpd
import logging
import numpy as np
import os
import pandas as pd
import pickle


"""
===============================================================
 Variables
===============================================================
"""


_report_template = """
<!doctype html>
<html lang="en">

    <head>
        <meta charset="utf-8">
        <title>Report: {{ title }}</title>
        <meta name="description" content="Random forest summary metrics for {{ title }}">
        <style>
            h1{
                font-size: 300%;
                text-align: center;
                padding: 30px;
                border-bottom: 5px double black;
            }
            h2{
                font-size: 200%;
                text-align: center;
                padding: 10px;
                padding-top: 50px;
                border-bottom: 1px solid black;
            }
            p {
                text-align: center;
            }
            table{
                border: 0;
                text-align: left;
                font-size: 120%;
                padding: 10px;
                margin-left:auto; 
                margin-right:auto;
            }
            th{
                border-bottom: 1px solid black;
                border-collapse: collapse;
                padding: 10px;
            }
            td{
                padding: 5px;
                padding-left: 10px;
            }
            img{
                height: 100vh;
                display: block;
                margin-left: auto;
                margin-right: auto;
                padding: 30px;
            }
        </style>
    </head>

    <body>
    
        <h1>Summary Report: {{ exp_title }}</h1>
        
        <h2>Geospatial Semantic Features</h2><br>
        {{ cd_plot }}
        
        <h2>Multicollinearity Reduction</h2><br>
        {{ ocorr_table }}
        {{ ocorr_plot }}
        
        <h2>Parameter Optimization</h2><br>
        {{ grid_table }}
        
        <h2>Cross Validation Performance</h2><br>
        {{ cv_plot }}
        
        <h2>Class Probabilities</h2><br>
        {{ prob_plot }}
        
        <h2>Feature Importance</h2><br>
        {{ imp_plot }}
        
        <h2>Outliers</h2><br>
        {{ outlier_plot }}
        
    </body>

</html>
"""


"""
===============================================================
 Configuration Functions
===============================================================
"""


def _global_config(config, settings=None, avail=None):
    """
     _global_config: obj -> obj
     
    ---------------------------------------------------------------
     
     Sets the local [config] to the global [settings] if a global
     setting exists. If all local [config] has been set, this function
     returns the original [config] or if no [settings] and [avail] settings
     exist.
     
     Required Parameters
     -------------------
     * config: obj
             The configobj instance object to scan if defaults
             have been customized.
             
     Optional Parameters
     -------------------
     * settings: None OR obj
             The global settings to use if it exists, otherwise
             use the defaults.
     * avail: None OR (listof str)
             The available [config] keys to set. Only these keys
             will be set to global defaults if they exist.
    
     Returns
     -------
     * config: obj
             The configobj instance after defaults have been set
             if applicable.
     
    ---------------------------------------------------------------
    """
    if settings is not None and avail is not None:
        for k in settings.keys():
            if k not in config and k in avail:
                config[k] = settings[k]
    return config

def analysis_config(config, settings=None):
    """
     analysis_config: obj -> obj
     
    ---------------------------------------------------------------
     
     Sets the defaults for a custom analysis configuration object
     from configobj.
     
     The defaults are only set if the setting does not exist (thus,
     it is implied that the user needs a default).
     
     Required Parameters
     -------------------
     * config: obj
             The configobj instance object to scan if defaults
             have been customized.
             
     Optional Parameters
     -------------------
     * settings: None
             The global settings to use if it exists, otherwise
             use the defaults.
    
     Returns
     -------
     * config: obj
             The configobj instance after defaults have been set
             if applicable.
     
    ---------------------------------------------------------------
    """
    config = _global_config(config, settings, ['cross_validation_tests', 'high_correlations', 'outlier_value', 'persist'])
    config['cross_validation_tests'] = [2, 5, 10] if 'cross_validation_tests' not in config else config['cross_validation_tests']
    config['high_correlations'] = [-0.7, 0.7] if 'high_correlations' not in config else config['high_correlations']
    config['outlier_value'] = 10 if 'outlier_value' not in config else config['outlier_value']
    config['persist'] = True if 'persist' not in config else config['persist']
    return config
    
    
def forest_config(config, n_jobs=[-1], settings=None):
    """
     forest_config: obj -> obj
     
    ---------------------------------------------------------------
     
     Sets the defaults for a custom forest configuration object
     from configobj.
     
     The defaults are only set if the setting does not exist (thus,
     it is implied that the user needs a default).
     
     Required Parameters
     -------------------
     * config: obj
             The configobj instance object to scan if defaults
             have been customized.
             
     Optional Parameters
     -------------------
     * settings: None
             The global settings to use if it exists, otherwise
             use the defaults.
    
     Returns
     -------
     * config: obj
             The configobj instance after defaults have been set
             if applicable.
     
    ---------------------------------------------------------------
    """
    config = _global_config(config, settings, RandomForestClassifier._get_param_names())
    config['n_estimators'] = [10, 64, 96, 128] if 'n_estimators' not in config else config['n_estimators']
    config['criterion'] = ['entropy'] if 'criterion' not in config else config['criterion']
    config['oob_score'] = [True] if 'oob_score' not in config else [True]
    config['class_weight'] = ['balanced'] if 'class_weight' not in config else config['class_weight']
    config['n_jobs'] = [n_jobs] if 'n_jobs' not in config else config['n_jobs']
    return config
    
    
def experiment_config(config):
    """
     experiment_config: obj -> obj
     
    ---------------------------------------------------------------
     
     Sets the defaults for a custom experiment info configuration object
     from configobj.
     
     The defaults are only set if the setting does not exist (thus,
     it is implied that the user needs a default).
     
     Required Parameters
     -------------------
     * config: obj
             The configobj instance object to scan if defaults
             have been customized.
    
     Returns
     -------
     * config: obj
             The configobj instance after defaults have been set
             if applicable.
     
    ---------------------------------------------------------------
    """
    config['title'] = '' if 'title' not in config else config['title']
    config['filter'] = [] if 'filter' not in config else config['filter']
    config['id'] = [] if 'id' not in config else config['id']
    config['keep_columns'] = [] if 'keep_columns' not in config else config['keep_columns']
    config['epsg'] = '4326' if 'epsg' not in config else config['epsg']
    config['units'] = 'units' if 'units' not in config else config['units']
    return config
    
    
def plot_config(config, settings=None):
    """
     plot_config: obj -> obj
     
    ---------------------------------------------------------------
     
     Sets the defaults for a custom experiment plot configuration object
     from configobj.
     
     The defaults are only set if the setting does not exist (thus,
     it is implied that the user needs a default).
     
     Required Parameters
     -------------------
     * config: obj
             The configobj instance object to scan if defaults
             have been customized.
             
     Optional Parameters
     -------------------
     * settings: None
             The global settings to use if it exists, otherwise
             use the defaults.
    
     Returns
     -------
     * config: obj
             The configobj instance after defaults have been set
             if applicable.
     
    ---------------------------------------------------------------
    """
    config = _global_config(config, settings)
    config['plot_style'] = 'whitegrid' if 'plot_style' not in config else config['plot_style']
    config['plot_color'] = 'gray' if 'plot_color' not in config else config['plot_color']
    config['plot_dpi'] = 300 if 'plot_dpi' not in config else config['plot_dpi']
    config['plot_ext'] = '.png' if 'plot_ext' not in config else config['plot_ext']
    return config
    
    
def settings_config(config):
    """
     settings_config: obj -> obj
     
    ---------------------------------------------------------------
     
     Sets the defaults for a custom settings configuration object
     from configobj.
     
     The defaults are only set if the setting does not exist (thus,
     it is implied that the user needs a default).
     
     Required Parameters
     -------------------
     * config: obj
             The configobj instance object to scan if defaults
             have been customized.
    
     Returns
     -------
     * config: obj
             The configobj instance after defaults have been set
             if applicable.
     
    ---------------------------------------------------------------
    """
    
    # (Settings) Configure the global settings
    settings = config['settings']
    config['settings']['cores'] = -1 if 'cores' not in settings else settings['cores']
    
    # (Plots) Configure global plot settings
    config['settings']['plot'] = {} if 'plot' not in settings else settings['plot']
    config['settings']['plot'] = plot_config(config['settings']['plot'])
    
    # (Analysis) Configure global analysis settings
    config['settings']['analysis'] = {} if 'analysis' not in settings else settings['analysis']
    config['settings']['analysis'] = analysis_config(config['settings']['analysis'])
    
    # (Forest) Configure global forest settings
    config['settings']['forest'] = {} if 'forest' not in settings else settings['forest']
    config['settings']['forest'] = forest_config(config['settings']['forest'], n_jobs=config['settings']['cores'])
    logging.info('Checked configuration file with defaults set when applicable')
    return config
    
    
"""
===============================================================
 Functions
===============================================================
"""

   
def gen_contrib(pkl, rf, outliers, variables, suspect_value=10, outlier_col='outlier_measure', cls_col='class', persist=True):
    """
     gen_contrib: str obj pd.DataFrame pd.DataFrame float str str bool -> pd.DataFrame
     
    ---------------------------------------------------------------
     
     Generates the contributions for each outlier given a [suspect] value. 
     
     Required Parameters
     -------------------
     * pkl: str
             The pickle file to store the probabilities
     * rf: obj
             The sklearn random forest model that has been previously trained.
     * outliers: pd.DataFrame
             The outlier measures obtained from the [rf] model from sklearn.
             It consists of two columns: [outlier_col] and [cls_col].
     * variables: pd.DataFrame
             The variables used to train the [rf] model from sklearn.
     * suspect_value: float
             The cutoff range to suspect an outlier. Any outlier measure
             greater than this value is considered an outlier.
     * outlier_col: str
             The outlier measure column name of [outliers].
     * cls_col: str
             The class column name of [outliers].
     
     Optional Parameters
     -------------------
     * persist: bool
             Whether to generate a pickle file or not.
             
     Returns
     -------
     * df: pd.DataFrame
             The result dataframe with the classes, and the variable
             contributions for each outlier.
             
    ---------------------------------------------------------------
    """
    if not os.path.isfile(pkl) and persist:
        
        # (Suspects) Obtain suspecting outliers
        suspects = pd.concat([outliers, variables], axis=1)
        suspects = suspects[suspects[outlier_col] > suspect_value]
        svariables = suspects.drop([outlier_col, cls_col], axis=1)  # variables of outliers
        
        # (Feat_Contrib) Obtain variable contributions to assigned class
        fc = ti.predict(rf, svariables.values)[2]
        contrib = []
        for c, cls in zip(fc, outliers[cls_col]):
            idx = np.where(rf.classes_ == cls)
            fci = [ft[idx][0] for ft in c]
            contrib.append(fci)
            
        # (Contrib_DF) Build informative contribution dataframe
        df = pd.DataFrame(contrib)
        df.columns = svariables.columns
        df.index = svariables.index
        df = pd.concat([suspects[cls_col], df], axis=1)
        with open(pkl, 'wb') as f:
            pickle.dump(df, f)
        logging.info('Pickled outlier variable contributions ' + pkl)
    else:
        with open(pkl, 'rb') as f:
            df = pickle.load(f)
        logging.info('Pickled outlier variable contributions already exists, skipping ' + pkl)
    return df
    
    
def gen_csv(out, df, persist=True, *args, **kwargs):
    """
     gen_csv: str obj bool *args **kwargs -> None
     
    ---------------------------------------------------------------
     
     Generates a csv file from a pandas [df] object. Skips
     the generation if the csv file already exists.
     
     Required Parameters
     -------------------
     * out: str
             The path to store the csv file with extension
     * df: obj
             A pandas dataframe to save
     * *args: *args
             Arguments to be passed to to_csv from pandas
     * **kwargs: **kwargs
             Keyword arguments to be passed to to_csv from pandas
             
     Optional Parameters
     -------------------
     * persist: bool
             Whether to regenerate a pickle file or not.
             
    ---------------------------------------------------------------
    """
    if not os.path.isfile(out):
        df.to_csv(out, *args, **kwargs)
        logging.info('Table saved at ' + out)
    else:
        logging.info('Table already exists, skipping ' + out)
        
        
def gen_f1_scores(pkl, obj, variables, targets, cv_files, cvs, persist=True, n_jobs=-1):
    """
     gen_f1_scores: str obj pd.DataFrame pd.Series (listof str) (listof int) bool int -> pd.DataFrame
     
    ---------------------------------------------------------------
     
     Generates the f1 scores for each cross validation test 
     specified by [cvs].
     
     Required Parameters
     -------------------
     * pkl: str
             The pickle file to store the probabilities
     * obj: obj
             The sklearn model that has been previously trained.
     * variables: pd.DataFrame
             The variables used to train the [obj] model from sklearn.
     * targets: pd.DataFrame
             The true target classes used to train the [obj] model from sklearn.
     * cv_files: (listof str)
             The cv files to save each cross_val_score object from
             sklearn.
     * cvs: (listof int)
             The cross validation folds for each test in list form.
     
     Optional Parameters
     -------------------
     * persist: bool
             Whether to generate a pickle file or not.
     * n_jobs: int
             Number of cores to use for parallel processing.
             
     Returns
     -------
     * cv_scores: pd.DataFrame
             The result dataframe with a column for the folds
             used for each cross validation and the respective
             mean f1 scores.
             
    ---------------------------------------------------------------
    """
    if not os.path.isfile(pkl) and persist:
        scores = []
        for cv_pkl, cv in zip(cv_files, cvs):
            f1_scores = gen_pkl(cv_pkl, _func=cross_val_score, _persist=persist, estimator=obj, X=variables.values, y=targets.values, cv=cv, scoring='f1_weighted', n_jobs=n_jobs)
            scores.append(f1_scores.mean())
        cvs = pd.Series(cvs, name='cv_folds')
        scores = pd.Series(scores, name='mean_f1_score')
        cv_scores = pd.concat([cvs, scores], axis=1)
        with open(pkl, 'wb') as f:
            pickle.dump(cv_scores, f)
        logging.info('Pickled F1 scores of cross validation tests ' + pkl)
    else:
        with open(pkl, 'rb') as f:
            cv_scores = pickle.load(f)
        logging.info('Pickled F1 scores of cross validation tests already exists, skipping ' + pkl)
    return cv_scores
    
    
def gen_gdc(data_files, target, epsg, pkl, cols=[], persist=True):
    """
     gen_gdc: (listof str) str str bool -> pd.DataFrame
     
    ---------------------------------------------------------------
     
     Reads the list of files containing geodata and combines
     them into one dataframe, before pickling them into a file
     at [pkl]. The data will also be projected to [epsg] and
     is assumed to all have the same coordinate reference system.
     
     Geometric variables such as geom_type, length, area (units^2), vertices,
     repx, and repy will also be included. Only the target variable
     will be included from the data files for classification.
     
     Required Parameters
     -------------------
     * data_files: (listof str)
             The geodata files to be read by geopandas via fiona.
             See http://www.gdal.org/ogr_formats.html
     * target: str
             The classification col in [gdc] with class data
     * epsg: str
             The coordinate reference system number in epsg to project the data to.
             http://geopandas.org/user.html#GeoSeries.to_crs
     * pkl: str
             The pickle file path the save the read geodata
             
     Optional Parameters
     -------------------
     * cols: (listof str)
             The list of column names to keep.
     * col_index: str OR None
             The unique id column to use as the index.
     * persist: bool
             Whether to generate a pickle file or not.
    
     Returns
     -------
     * gd: pd.DataFrame
             The combined data from [data_files] projected
             to [epsg]
     
    ---------------------------------------------------------------
    """
    if not os.path.isfile(pkl) and persist:
        gdc = helpers.concat_gdf(data_files, epsg=epsg)
        crs = gdc.crs
        variables = helpers.get_series(gdc, series_cols=cols + ['geom_type', 'length', 'area'])
        variables['area'] = variables['area'].apply(sqrt)
        vtx = helpers.get_vtx(gdc)
        pts = gdc.representative_point()
        pts = pd.DataFrame([[p.x, p.y] for p in pts], columns=['repx', 'repy'])
        gdc = pd.concat([gdc[target], pts, variables, vtx, gdc.geometry], axis=1)
        gdc = gpd.GeoDataFrame(gdc)
        gdc.crs = crs
        with open(pkl, 'wb') as f:
            pickle.dump(gdc, f)
        logging.info('Pickled GeoDataFrame file ' + pkl)
    else:
        with open(pkl, 'rb') as f:
            gdc = pickle.load(f)
        logging.info('GeoDataFrame file exists, skipping pickle for ' + pkl)
    return gdc
    
    
def gen_gdcn(gdc,
             gdn,
             target,
             pkl,
             gdn_ipattern='near_',
             corr_pkl=None,
             corr_range=(-0.8, 0.8),
             ignr_corr=None,
             scaler=None,
             ignr_scale=None,
             persist=True):
    """
     gen_gdcn: gpd.GeoDataFrame
               gpd.GeoDataFrame
               str
               str
               str OR None
               str OR None
               (tupleof num)
               (listof str) OR None
               obj
               (listof str) OR None
               bool
               -> pd.DataFrame
     
    ---------------------------------------------------------------
     
     Combines the relationship data [gdn] with [gdc]. Also performs
     preprocessing of multicollinearity reduction, removal of 0 variance
     variables, and scaling depending on [corr_pkl]..[scaler] arguments.
     
     Required Parameters
     -------------------
     * gdf: gpd.GeoDataFrame
             The geodataframe with the geometric variables and the original
             data used to generate [gdn]
     * gdn: gpd.GeoDataFrame
             The geodataframe with the nearest distance to each
             [target] class of [gdc] for each row of [gdc]
     * target: str
             The group col in [gdc] representing the classification groups
     * pkl: str
             The pickle file path the save the combined variables data
             
     Optional Parameters
     -------------------
     * gdn_ipattern: str OR None
             If not None, set this to the alias for the [gdn] variables
             pattern in which each column corresponds to a unique class in the 
             [target] column with an added alias in front of its name.
             E.g. If gdn_ipattern = 'near_' and a class from target is 'bus_stop',
             the corresponding target class col from [gdn] would be 'near_bus_stop'
             Once set, this will order the [gdn] columns in descending order
             of [target] class counts - thus the class with the most counts are
             first and the the class with the least counts are last. This is
             useful for the ordered reduction of multicollinearity included
             with this function.
     * corr_pkl: str OR None
             If not None, reduces multicollinearity in the data by only
             limiting to variables that are not correlated to each
             other. This considers variables to keep in order starting
             with variables from the [gdf] then variables from the [gdn].
             Specify a path to pickle the details of the correlation
             removal in order to apply it.
     * corr_range: (tupleof num)
             If [corr_pkl] is not None, specify the negative (1st item)
             and positive (2nd item) correlation thresholds to considering
             multicollinearity.
     * ignr_corr: (listof str) OR None
             If [corr_pkl] is not None, specify the columns to ignore
             when checking for high correlation removal
     * scaler: obj OR None
             If not None, uses a sklearn scaler object to scale the
             non-categorical variables.
     * ignr_scale: (listof str) OR None
             If [scaler] is not None, specify the columns to ignore
             when scaling variables.
     * persist: bool
             Whether to regenerate a pickle file or not.
    
     Returns
     -------
     * gdcn: pd.DataFrame
             The [gdc] data with the added relationship variables
             modified with preprocessing from [corr_pkl]..[scaled]
             adjustments if applicable.
     
    ---------------------------------------------------------------
    """
    if not os.path.isfile(pkl) or not persist:
        gdn = gdn['near_' + gdc[target].value_counts().index] if gdn_ipattern is not None else gdn  # order by freq of [target]
        gdcn = pd.concat([gdc, gdn], axis=1)
        
        # (Variance) Remove zero variance variables
        var = gdcn.var()
        gdcn = gdcn.drop(var[var == 0].index, axis=1)
        
        # (Multicollinearity) Remove colinear variables in order
        if corr_pkl is not None:
            if ignr_corr is None:
                ocorr = helpers.ocorr_df(gdcn.drop(target, axis=1), corr_range[0], corr_range[1])
                vkeep = ocorr.keep.tolist() + [target]
            else:
                corr_cols = [c for c in gdcn.columns if c not in ignr_corr and c != target]
                corr_chk = gdcn.drop(target, axis=1)[corr_cols]
                ocorr = helpers.ocorr_df(corr_chk, corr_range[0], corr_range[1])
                vkeep = ocorr.keep.tolist() + [target] + ignr_corr
            gdcn = gdcn[vkeep]  # keep non-correlated variables          
            with open(corr_pkl, 'wb') as f:
                pickle.dump(ocorr, f)
            logging.info('Pickled dictionary of removed correlated variables at ' + corr_pkl)
        
        # (Scale) Use a scaler to transform variables
        if scaler is not None:
            if ignr_scale is None:
                scale_cols = gdcn.columns
            else:
                scale_cols = [c for c in gdcn.columns if c not in ignr_scale]
            gdcn[scale_cols] = scaler.fit_transform(gdcn[scale_cols].values)
            
        # (Save) Pickle the [complete] data
        with open(pkl, 'wb') as f:
            pickle.dump(gdcn, f)
        logging.info('Calculated and pickled combined geodata file ' + pkl)
    else:
        with open(pkl, 'rb') as f:
            gdcn = pickle.load(f)
        logging.info('Combined geodata already calculated, skipping pickle for ' + pkl)
    return gdcn
    
    
def gen_html_plot(_fig, *args, **kwargs):
    """
     gen_html_plot: obj -> str
     
    ---------------------------------------------------------------
     
     Converts a matplotlib figure [obj] to bytes for use in
     data uri of html templates. Original code modified from
     [1].
     
     References
     ----------
     * [1] http://stackoverflow.com/questions/31492525/converting-matplotlib-png-to-base64-for-viewing-in-html-template
     
     Required Parameters
     -------------------
     * _fig: obj
             A matplotlib figure obj.
     
     Optional Parameters
     -------------------
     * *args: *args
             Arguments to be passed to [fig].savefig
     * **kwargs: **kwargs
             Keyword arguments to be passed to [fig].savefig
             
     Returns
     -------
     * html_plot: str
             The string representation of image data from [fig]
             to be embedded as a data uri in an html template.
     
    ---------------------------------------------------------------
    """
    fig_io = BytesIO()
    _fig.savefig(fig_io, *args, **kwargs)
    fig_io.seek(0)
    data_uri = base64.b64encode(fig_io.getvalue()).decode('utf8')
    html_plot = '<img src="data:image/png;base64,' + data_uri + '"\>'
    return html_plot
    
    
def gen_imp(pkl, obj, variable_cols, persist=True):
    """
     gen_imp: str obj (listof str) bool -> pd.DataFrame
     
    ---------------------------------------------------------------
     
     Uses a trained model [obj] from sklearn to extract the
     variable importances.
     
     Required Parameters
     -------------------
     * pkl: str
             The pickle file to store the variable importances.
     * obj: obj
             The sklearn model that has been previously trained.
     * variable_cols: pd.DataFrame
             The names of the variables used to train the [obj] model
             from sklearn in order.
     
     Optional Parameters
     -------------------
     * persist: bool
             Whether to generate a pickle file or not.
             
     Returns
     -------
     * rf_imp: pd.DataFrame
             The variable importance dataframe.
     
    ---------------------------------------------------------------
    """
    if not os.path.isfile(pkl) or not persist:
        imp = pd.DataFrame(obj.feature_importances_, columns=['importance'], index=variable_cols)
        imp['variable'] = imp.index.values
        imp = imp.sort_values(by='importance', ascending=False)
        with open(pkl, 'wb') as f:
            pickle.dump(imp, f)
        logging.info('Pickled random forest variable importances ' + pkl)
    else:
        with open(pkl, 'rb') as f:
            imp = pickle.load(f)
        logging.info('Pickled random forest variable importances already exists, skipping ' + pkl)
    return imp
    
    
def gen_mprob(pkl, prob, cls_col='predict', prob_col='max_prob', persist=True):
    """
     gen_mprob: str pd.DataFrame str str bool -> pd.DataFrame
     
    ---------------------------------------------------------------
     
     Obtains the mean probability for each class given the generated
     probabilities.
     
     Required Parameters
     -------------------
     * pkl: str
             The pickle file to store the mean class probabilities.
     * prob: pd.DataFrame
             The probabilities to calculate the mean class probabilities
             from. There must be a class column named [target] and 
             a probability column named [prob_col].
             
     Optional Parameters
     -------------------
     * cls_col: str
             The class column name from [prob].
     * prob_col: str
             The probability column name from [prob].
     * persist: bool
             Whether to generate a pickle file or not.
             
     Returns
     -------
     * mprob: pd.DataFrame
             The dataframe with information on the mean probabilities
             for each class sorted from largest to smallest probability.
     
    ---------------------------------------------------------------
    """
    if not os.path.isfile(pkl) or not persist:
        mprob = pd.DataFrame(prob.groupby(cls_col)[prob_col].mean())
        mprob[cls_col] = mprob.index.values
        mprob = mprob.sort_values(by=prob_col, ascending=False)
        with open(pkl, 'wb') as f:
            pickle.dump(mprob, f)
        logging.info('Pickled mean class probabilities ' + pkl)
    else:
        with open(pkl, 'rb') as f:
            mprob = pickle.load(f)
        logging.info('Pickled mean class probabilities already exists, skipping ' + pkl)
    return mprob
    
    
def gen_outliers(pkl, prox_files, target_cls, persist=True):
    """
     gen_outliers: str (listof str) pd.Series bool -> pd.DataFrame
     
    ---------------------------------------------------------------
     
     Obtains the class outlier measures for each instance of data
     using proximities as described by [1].
     
     References
     ----------
     * [1] Breiman, Leo: https://www.stat.berkeley.edu/~breiman/Using_random_forests_v4.0.pdf
     
     Required Parameters
     -------------------
     * pkl: str
             The pickle file to store the mean proximities.
     * prox_files: (listof str)
             The joblib pickle files with the stored proximities.
             Each file represents a proximity matrix for a class
             in the same order as [target_cls].
     * target_cls: pd.Series
             The series of classes to generate the mean proximities on.
             Each class must have a corresponding [prox_files] in order.
     
     Optional Parameters
     -------------------
     * persist: bool
             Whether to generate a pickle file or not.
             
     Returns
     -------
     * outliers: pd.DataFrame
             The dataframe of outlier measures and the true classes
             for each instance of data.
     
    ---------------------------------------------------------------
    """
    if not os.path.isfile(pkl) or not persist:
        iprox = []  # within-class mean prox of each instance
        icls = []  # classes of each instance
        idx = []  # original instance indices
        for prox_pkl, cls in zip(prox_files, target_cls):
            prox_df = joblib.load(prox_pkl)
            prox = prox_df.values
            np.fill_diagonal(prox, np.nan)  # set matching instances to nan
            out_n = len(prox) / np.nansum(prox**2, axis=0)  # outlier measure of instances n
            iout = (out_n - np.median(out_n)) / mad(out_n, center=np.median)  # normalized outlier measure
            iprox = iprox + list(iout)
            icls = icls + [cls] * len(prox)
            idx = idx + list(prox_df.index.values)
        iprox = pd.Series(iprox, name='outlier_measure')
        icls = pd.Series(icls, name='class')
        outliers = pd.concat([iprox, icls], axis=1)
        outliers.index = idx
        with open(pkl, 'wb') as f:
            pickle.dump(outliers, f)
        logging.info('Pickled outlier measures ' + pkl)
    else:
        with open(pkl, 'rb') as f:
            outliers = pickle.load(f)
        logging.info('Pickled outlier measures already exists, skipping ' + pkl)
    return outliers
    
    
def gen_pkl(_pkl, _func, _lib='pickle', _persist=True, *args, **kwargs):
    """
     gen_pkl: (listof str) function str bool *args **kwargs -> any
     
    ---------------------------------------------------------------
     
     Generates a pickled file from data returned from [-func] after
     passing [*args] and/or [**kwargs].
     
     Required Parameters
     -------------------
     * _pkl: str
             The path to store the pickled file
     * _func: function
             A function that returns data to be pickled
             
     Optional Parameters
     -------------------
     * _lib: str
             An object that loads and dumps pickle files
             from data returned from [_func]. Currently
             supported inputs are 'pickle' and 'joblib'.
     * persist: bool
             Whether to regenerate a pickle file or not.
     * *args: *args
             Arguments to be passed to [_func]
     * **kwargs: **kwargs
             Keyword arguments to be passed to [_func]
    
     Returns
     -------
     * data: any
             The return value from [_func] after passing
             [*args] and [**kawargs].
     
    ---------------------------------------------------------------
    """
    if _lib not in ['pickle', 'joblib']:
        raise(Exception('Error: ' + _lib + ' is not a supported object for load and dump.'))
    if not os.path.isfile(_pkl) or not _persist:
        data = _func(*args, **kwargs)
        if _lib == 'pickle' and _persist:
            with open(_pkl, 'wb') as f:
                pickle.dump(data, f)
        elif _lib == 'joblib' and _persist:
            joblib.dump(data, _pkl)
        logging.info('Pickled data from ' + _func.__name__ + ' for ' + _pkl)
    else:
        if _lib == 'pickle':
            with open(_pkl, 'rb') as f:
                data = pickle.load(f)
        elif _lib == 'joblib':
            data = joblib.load(_pkl)
        logging.info('Pickled data from ' + _func.__name__ + ' already exists, skipping ' + _pkl)
    return data
    
    
def gen_prob(pkl, obj, variables, persist=True):
    """
     gen_prob: str obj pd.DataFrame bool -> pd.DataFrame
     
    ---------------------------------------------------------------
     
     Uses a trained model [obj] from sklearn to extract the
     probabilities for each class, the maximum probability of the
     predicted class, and the predicted class information given
     the attributes of predict_proba and classes_, and method
     of predict.
     
     Required Parameters
     -------------------
     * pkl: str
             The pickle file to store the probabilities
     * obj: obj
             The sklearn model that has been previously trained.
     * variables: pd.DataFrame
             The variables used to train the [obj] model from sklearn.
     
     Optional Parameters
     -------------------
     * persist: bool
             Whether to generate a pickle file or not.
             
     Returns
     -------
     * prob: pd.DataFrame
             The probability dataframe with information on the
             probabilities for each class, the maximum probabilty
             for the predicted class, and the predicted class
             itself in the respective order.
     
    ---------------------------------------------------------------
    """
    if not os.path.isfile(pkl) or not persist:
        pred = pd.Series(obj.predict(variables.values), name='predict')
        cls_prob = pd.DataFrame(obj.predict_proba(variables.values), columns=obj.classes_)
        max_prob = pd.Series(cls_prob.apply(max, axis=1).values, name='max_prob')
        prob = pd.concat([cls_prob, max_prob, pred], axis=1)
        with open(pkl, 'wb') as f:
            pickle.dump(prob, f)
        logging.info('Pickled random forest probabilities ' + pkl)
    else:
        with open(pkl, 'rb') as f:
            prob = pickle.load(f)
        logging.info('Pickled random forest probabilities already exists, skipping ' + pkl)
    return prob
    
    
def gen_prox(pkl, obj, variables, persist=True):
    """
     gen_prox: str obj pd.DataFrame str bool
     
    ---------------------------------------------------------------
     
     Uses a trained model [obj] from sklearn to extract the
     proximities for each [variables] and saves it to a [pkl].
     
     This function is designed for parallel processing and reduction
     of memory for large datasets, and thus does not return data. To
     retrieve the results, load the data from the file at [pkl] using
     joblib.load.
     
     Required Parameters
     -------------------
     * pkl: str
             The pickle file to store the proximities. This is created
             using joblib.
     * obj: obj
             The sklearn model that has been previously trained.
     * variables: pd.DataFrame
             The training variables matching the [obj] model from sklearn.
     
     Optional Parameters
     -------------------
     * persist: bool
             Whether to generate a pickle file or not.
     
    ---------------------------------------------------------------
    """
    if not os.path.isfile(pkl) or not persist:
        prox = 1. - helpers.rf_prox(obj, variables.values)
        prox = pd.DataFrame(prox, index=variables.index)
        with open(pkl, 'wb') as f:
            pickle.dump(prox, f)
        logging.info('Pickled random forest proximities ' + pkl)
    else:
        logging.info('Pickled random forest proximities already exists, skipping ' + pkl)
        
        
def gen_rfg(rfg_files, grid, variables, targets, persist=True):
    """
     gen_rfg: (listof str) obj pd.DataFrame pd.Series bool -> pd.DataFrame
     
    ---------------------------------------------------------------
     
     Trains a random forest classifier for each [grid] parameter combination and
     returns a dataframe that summarizes its oob score, fit,
     and the path to the stored pickle files.
     
     Required Parameters
     -------------------
     * rft_files: (listof str)
             The list of pickle files to save each random forest
     * grid: obj
             The parameter grid to generate random forests combinations on.
     * variables: pd.DataFrame
             The variables to use for training the random forest classifier
     * targets: pd.Series
             The prediction targets to use for training the random forest
             classifier
             
     Optional Parameters
     -------------------
     * persist: bool
             Whether to generate a pickle file or not.
             
     Returns
     -------
     * : pd.DataFrame
             The summary dataframe consisting of the number of trees
             used for experimentation, out of bag score, score (fit)
             of the random forest model and a the pickled file
             that the random forest model is stored in.
     
    ---------------------------------------------------------------
    """
    rfg_oob = []
    grid_names = list(list(grid)[0].keys())
    for pkl, g in zip(rfg_files, grid):
        if not os.path.isfile(pkl) or not persist:
            rfg = RandomForestClassifier(**g)
            rfg = gen_pkl(pkl, _func=rfg.fit, _lib='joblib', _persist=persist, X=variables.values, y=targets.values)
        else:
            rfg = joblib.load(pkl)
            logging.info('Pickled random forest grid already exists, skipping ' + pkl)
        rfg_oob.append(list(g.values()) + [1 - rfg.oob_score_, rfg.score(variables.values, targets.values), pkl])
    return pd.DataFrame(rfg_oob, columns=grid_names + ['oob_error', 'score', 'pkl'])
    
    