"""
 main.py
 Richard Wen (rwenite@gmail.com)

===============================================================

 A script for running random forests for pattern recognition
 of spatial data.
 
 Call python via console:
   python thesis.py config.txt path\\to\\workspace_folder
   
===============================================================
"""


"""
===============================================================
 Modules
===============================================================
"""


from configobj import ConfigObj
from jinja2 import Template
from joblib import Parallel, delayed
from modules import files, helpers, logs, workflow
from sklearn.grid_search import ParameterGrid
from sklearn.externals import joblib


import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import seaborn as sns
import sys


"""
===============================================================
 Script
===============================================================
"""


if __name__ == "__main__":
    config_file = sys.argv[1]
    config = ConfigObj(config_file, unrepr=True)
    config = workflow.settings_config(config)
                        
    # (Execution) Execute automated script
    # ---------------------------------------------------------------
    
    # (Read_Config) Read the settings into appropriate variables    
    settings = config.pop('settings')
    experiments = config
    
    # (Run_Experiments) Run each experiment using user settings
    settings['workspace'] = sys.argv[2] if len(sys.argv) > 2 else settings['workspace']   
    plt.rcParams.update({'figure.autolayout': True})  # tight layout for matplotlib
    files.add_folder(settings['workspace'])
    for alias, info in experiments.items():
        
        # (Setup)
        # ---------------------------------------------------------------
        
        # (Create_Subfolder) Subfolder for processing each project
        project_workspace = os.path.join(settings['workspace'], alias)
        memory_subfolder = os.path.join(project_workspace, 'memory')
        files.add_folder(project_workspace)
        files.add_folder(memory_subfolder)
        
        # (Create_Log) Setup logging file for data source
        log_file = os.path.join(project_workspace, alias + '_log.csv')
        logger = logs.save_log(log_file, 'a')
        logging.info('Created log file ' + log_file)
        
        # (Data_Paths) Obtain the data paths and filter out files
        data_local = files.Fileset(info['src'], memory_subfolder, clean=True, overwrite=False)
        data_files = [p for p in data_local.paths for f in info['filter'] if f in p] if info['filter'] != [] else data_local.paths
        
        # (Misc_Config) Obtain default info/analysis/forest settings and use local if exists
        info = workflow.experiment_config(info)
        analysis = info['analysis'] if 'analysis' in info else settings['analysis']
        analysis = workflow.analysis_config(analysis, settings=settings)
        forest = info['forest'] if 'forest' in info else settings['forest']
        forest = workflow.forest_config(forest, settings['cores'], settings=settings)
        plot = info['plot'] if 'plot' in info else settings['plot']
        plot = workflow.plot_config(plot, settings=settings)
        
        # (Process) Calculate data and train classifier
        # ---------------------------------------------------------------
        
        # (Read_GD) Read and pickle the geo [data_files] with geo vars
        gdc_pkl = os.path.join(memory_subfolder, alias + '_gdc.pkl')        
        gdc = workflow.gen_gdc(data_files, info['target'], info['epsg'], gdc_pkl, cols=info['id'] + info['keep_columns'], persist=analysis['persist'])
        gdc_cls = gdc[info['target']].unique()
        if info['id'] != []:  # get ids if needed
            gid = gdc[info['id']]
            gdc = gdc.drop(info['id'], axis=1)
        
        # (Near_Dist) Calculate the nearest distance for each class
        gdn_files = [os.path.join(memory_subfolder, alias + '_' + cls + '_nbd.pkl') for cls in gdc_cls]
        gdn = Parallel(n_jobs=settings['cores'])(delayed(workflow.gen_pkl)(pkl, _func=helpers.nb_dist, _persist=analysis['persist'], origin=gdc, near=gdc[gdc[info['target']] == cls], name='near_' + cls) for pkl, cls in zip(gdn_files, gdc_cls))
        gdn = pd.concat(gdn, axis=1)
        
        # (Combine_GD) Combine the [gdc] raw data with [gdn] nearest distances
        gdcn_pkl = os.path.join(memory_subfolder, alias + '_gdcn.pkl')
        ocorr_pkl = os.path.join(memory_subfolder, alias + '_ocorr.pkl')
        gdcn = workflow.gen_gdcn(gdc, gdn, info['target'], gdcn_pkl, corr_pkl=ocorr_pkl, corr_range=analysis['high_correlations'], persist=analysis['persist'])
        
        # (RForest) Generate random forest classifiers on [gdcn] empirically
        grid = ParameterGrid(forest)
        rfg_files = [os.path.join(memory_subfolder, alias + '_rfg' + str(i) + '.pkl') for i in range(0, len(grid))]
        rf_vars = gdcn.drop(info['target'], axis=1)
        rf_targets = gdcn[info['target']]
        rf_grid = workflow.gen_rfg(rfg_files, grid, rf_vars, rf_targets, persist=analysis['persist'])

        # (RForest_Select) Select random forest with lowest oob error
        rf_pkl = rf_grid[rf_grid['oob_error'] == rf_grid['oob_error'].min()]['pkl'].values[0]
        rf = joblib.load(rf_pkl)
        
        # (RForest_Score) Cross validate selected forest for performance measure
        cv_files = [os.path.join(memory_subfolder, alias + '_rf_cv' + str(cv) + '.pkl') for cv in analysis['cross_validation_tests']]
        cv_scores_pkl = os.path.join(memory_subfolder, alias + '_cvscores.pkl')
        cv_scores = workflow.gen_f1_scores(cv_scores_pkl, rf, rf_vars, rf_targets, cv_files, analysis['cross_validation_tests'], persist=analysis['persist'], n_jobs=settings['cores'])
        
        # (RForest_Prob) Obtain probabilities from chosen random forest
        rf_prob_pkl = os.path.join(memory_subfolder, alias + '_rf_prob.pkl')
        rf_prob = workflow.gen_prob(rf_prob_pkl, rf, rf_vars, persist=analysis['persist'])
        rf_mprob_pkl = os.path.join(memory_subfolder, alias + '_rf_mprob.pkl')
        rf_mprob = workflow.gen_mprob(rf_mprob_pkl, rf_prob)  # mean class prob
        
        # (RForest_Prox) Obtain proximities from chosen random forest
        rf_outliers_pkl = os.path.join(memory_subfolder, alias + '_rf_outliers.pkl')
        rf_prox_files = [os.path.join(memory_subfolder, alias + '_' + cls + '_rf_prox.pkl') for cls in gdc_cls]
        if not os.path.isfile(rf_outliers_pkl) or not analysis['persist']:
            Parallel(n_jobs=settings['cores'])(delayed(workflow.gen_prox)(pkl, rf, gdcn[gdcn[info['target']] == cls].drop(info['target'], axis=1), persist=analysis['persist']) for pkl, cls in zip(rf_prox_files, gdc_cls))
        rf_outliers = workflow.gen_outliers(rf_outliers_pkl, rf_prox_files, gdc_cls, persist=analysis['persist'])
        
        # (Var_Importance) Obtain variable importances for entire dataset
        rf_imp_pkl = os.path.join(memory_subfolder, alias + '_rf_imp.pkl')
        rf_imp = workflow.gen_imp(rf_imp_pkl, rf, rf_vars.columns, persist=analysis['persist'])
        
        # (Var_Contrib) Obtain variable contributions for outliers
        rf_contrib_pkl = os.path.join(memory_subfolder, alias + '_rf_contrib.pkl')
        rf_contrib = workflow.gen_contrib(rf_contrib_pkl, rf, rf_outliers, rf_vars, suspect_value=analysis['outlier_value'], persist=analysis['persist'])
        
        # (Tables) Create .csv files from pandas
        # ---------------------------------------------------------------
        tables_subfolder = os.path.join(project_workspace, 'tables')
        files.add_folder(tables_subfolder)
        
        # (Vars) variables table with geospatial semantic variables and included cols
        var_table_out = os.path.join(tables_subfolder, alias + '_vars.csv')
        var_table = gdc.merge(gid, 'left', left_index=True, right_index=True) if info['id'] is not None else gdc
        workflow.gen_csv(var_table_out, var_table, persist=analysis['persist'], index=False)
        
        # (Near_Vars) Nearest neighbour distance variables for each class
        near_table_out = os.path.join(tables_subfolder, alias + '_neardist_vars.csv')
        near_table = gdn.merge(gid, 'left', left_index=True, right_index=True) if info['id'] is not None else gdn
        workflow.gen_csv(near_table_out, near_table, persist=analysis['persist'], index=False)
        
        # (Corr_Remove_Table) Removed correlation details
        ocorr_table_out = os.path.join(tables_subfolder, alias + '_multicorr_reduction.csv')
        with open(ocorr_pkl, 'rb') as f:
            ocorr = pickle.load(f)
        ocorr['len'] = ocorr['remove'].apply(len)
        ocorr['remove'] = ocorr['remove'].apply(', '.join)
        ocorr['order'] = ocorr.index.values
        workflow.gen_csv(ocorr_table_out, ocorr, persist=analysis['persist'], index=False, headers=['Kept', 'Removed', 'Removed Sum'])
        
        # (Dataset) Full dataset used for training/testing random forest
        dataset_table_out = os.path.join(tables_subfolder, alias + '_rf_dataset.csv')
        dataset_table = gdcn.merge(gid, left_index=True, right_index=True) if info['id'] != [] else gdcn
        workflow.gen_csv(dataset_table_out, dataset_table, persist=analysis['persist'], index=False)
        
        # (Param_Grid) Parameter grid results
        grid_table_out = os.path.join(tables_subfolder, alias + '_param_optimize.csv')
        workflow.gen_csv(grid_table_out, rf_grid, persist=analysis['persist'], index=False)
        
        # (CVScores_Table) Cross validation test f1 scores
        cv_table_out = os.path.join(tables_subfolder, alias + '_f1scores.csv')
        workflow.gen_csv(cv_table_out, cv_scores, persist=analysis['persist'], index=False)
        
        # (Prob_Table) Random Forest Probabilities for each prediction
        prob_table_out = os.path.join(tables_subfolder, alias + '_prob.csv')
        prob_table = rf_prob.merge(gid, left_index=True, right_index=True) if info['id'] != [] else rf_prob
        workflow.gen_csv(prob_table_out, prob_table, persist=analysis['persist'], index=False)
        
        # (Outlier_Table) Random Forest Outlier measures for each class
        outlier_table_out = os.path.join(tables_subfolder, alias + '_outlier_measures.csv')
        outlier_table = rf_outliers.merge(gid, left_index=True, right_index=True) if info['id'] != [] else rf_outliers
        workflow.gen_csv(outlier_table_out, outlier_table, persist=analysis['persist'], index=False)
        
        # (FImportance_Table) variable Importances
        imp_table_out = os.path.join(tables_subfolder, alias + '_var_importances.csv')
        workflow.gen_csv(imp_table_out, rf_imp, persist=analysis['persist'], index=False)
        
        # (FContrib_Table) Variable contributions for outlier classes
        contrib_table_out = os.path.join(tables_subfolder, alias + '_outlierclass_contrib.csv')
        contrib_table = rf_contrib.merge(gid, left_index=True, right_index=True) if info['id'] != [] else rf_contrib
        workflow.gen_csv(contrib_table_out, contrib_table, persist=analysis['persist'], index=False)
        
        # (Plots) Create static plots from seaborn and matplotlib
        # ---------------------------------------------------------------
        sns.set(context='paper', style=plot['plot_style'])  # seaborn global style
        plots_subfolder = os.path.join(project_workspace, 'plots')
        files.add_folder(plots_subfolder)
        
        # (Setup_Plots) Setup folders and settings for plots
        var_plots_subfolder = os.path.join(plots_subfolder, 'variables')
        xy_plots_subfolder = os.path.join(plots_subfolder, 'xy')
        outlier_plots_subfolder = os.path.join(plots_subfolder, 'outliers')
        files.add_folder(var_plots_subfolder)
        files.add_folder(xy_plots_subfolder)
        files.add_folder(outlier_plots_subfolder)
        cls_title = info['target'].capitalize()
        data_title = info['title']
        dist_title = info['units'].capitalize()
        
        # (Class_Hist) Plot class distributions
        cd_plot_out = os.path.join(var_plots_subfolder, alias + '_class_hist') + plot['plot_ext']
        cd_plot_pkl = os.path.join(memory_subfolder, alias + '_class_hist.pkl')
        if not os.path.isfile(cd_plot_out) or not analysis['persist']:
            cd_plot_data = gdcn[info['target']].value_counts()
            fig = plt.figure(figsize=(8.5, 11))
            cd_plot = sns.barplot(x=cd_plot_data.values, y=cd_plot_data.index, color=plot['plot_color'])
            cd_plot.set_title(data_title + ': Class Counts')
            cd_plot.set_xlabel('Count')
            cd_plot.set_ylabel('Class')
            cd_plot.figure.savefig(cd_plot_out, dpi=plot['plot_dpi'])
            logging.info('Figure saved at ' + cd_plot_out)
            with open(cd_plot_pkl, 'wb') as f:
                pickle.dump(cd_plot, f)
            logging.info('Pickled figure at ' + cd_plot_pkl)
            plt.close('all')
        else:
            logging.info('Figure exists, skipping ' + cd_plot_out)
        
        # (Area_Hist) Plot the distribution of area sizes
        area_plot_out = os.path.join(var_plots_subfolder, alias + '_area_hist') + plot['plot_ext']
        area_plot_pkl = os.path.join(memory_subfolder, alias + '_area_hist.pkl')
        if not os.path.isfile(area_plot_out) or not analysis['persist']:
            fig = plt.figure(figsize=(4.75, 5.5))
            poly_areas = gdc['area']
            area_plot = sns.distplot(poly_areas, kde=False, color=plot['plot_color'], rug=True)
            area_plot.set_title(data_title + ': Area Distribution')
            area_plot.set_xlabel('Area (' + info['units'] + ' squared)')
            area_plot.set_ylabel('Count')
            area_plot.figure.savefig(area_plot_out, dpi=plot['plot_dpi'])
            logging.info('Figure saved at ' + area_plot_out)
            with open(area_plot_pkl, 'wb') as f:
                pickle.dump(area_plot, f)
            logging.info('Pickled figure at ' + area_plot_pkl)
            plt.close('all')
        else:
            logging.info('Figure exists, skipping ' + area_plot_out)
        
        # (Vertex_Hist) Plot the distribution of vertices
        vtx_plot_out = os.path.join(var_plots_subfolder, alias + '_vtx_hist') + plot['plot_ext']
        vtx_plot_pkl = os.path.join(memory_subfolder, alias + '_vtx_hist.pkl')
        if not os.path.isfile(vtx_plot_out) or not analysis['persist']:
            fig = plt.figure(figsize=(4.75, 5.5))
            vtx_plot = sns.distplot(gdc['vertices'], kde=False, color=plot['plot_color'], rug=True)
            vtx_plot.set_title(data_title + ': Vertices Distribution')
            vtx_plot.set_xlabel('Vertices')
            vtx_plot.set_ylabel('Count')
            vtx_plot.figure.savefig(vtx_plot_out, dpi=plot['plot_dpi'])
            logging.info('Figure saved at ' + vtx_plot_out)
            with open(vtx_plot_pkl, 'wb') as f:
                pickle.dump(vtx_plot, f)
            logging.info('Pickled figure at ' + vtx_plot_pkl)
            plt.close('all')
        else:
            logging.info('Figure exists, skipping ' + vtx_plot_out)
            
        # (Length_Hist) Plot the distribution of lengths
        len_plot_out = os.path.join(var_plots_subfolder, alias + '_len_hist') + plot['plot_ext']
        len_plot_pkl = os.path.join(memory_subfolder, alias + '_len_hist.pkl')
        if not os.path.isfile(len_plot_out) or not analysis['persist']:
            fig = plt.figure(figsize=(4.75, 5.5))
            len_plot = sns.distplot(gdc['length'], kde=False, color=plot['plot_color'], rug=True)
            len_plot.set_title(data_title + ': Length Distribution')
            len_plot.set_xlabel('Length')
            len_plot.set_ylabel('Count')
            len_plot.figure.savefig(len_plot_out, dpi=plot['plot_dpi'])
            logging.info('Figure saved at ' + len_plot_out)
            with open(len_plot_pkl, 'wb') as f:
                pickle.dump(len_plot, f)
            logging.info('Pickled figure at ' + len_plot_pkl)
            plt.close('all')
        else:
            logging.info('Figure exists, skipping ' + len_plot_out)
        
        # (Near_Var_Box) Plot variables with near distances of classes
        near_plot_out = os.path.join(var_plots_subfolder, alias + '_near_box') + plot['plot_ext']
        near_plot_pkl = os.path.join(memory_subfolder, alias + '_near_box.pkl')
        if not os.path.isfile(near_plot_out) or not analysis['persist']:
            fig = plt.figure(figsize=(8.5, 11))
            near_plot = sns.boxplot(data=gdn, orient='h', color=plot['plot_color'], fliersize=3, width=0.3, linewidth=0.5)
            near_plot.set_title(data_title + ': Distribution of First Nearest Class Distances')
            near_plot.set_xlabel('Distance in ' + dist_title)
            near_plot.set_ylabel('Nearest Class')
            near_plot.figure.savefig(near_plot_out, dpi=plot['plot_dpi'])
            logging.info('Figure saved at ' + near_plot_out)
            with open(near_plot_pkl, 'wb') as f:
                pickle.dump(near_plot, f)
            logging.info('Pickled figure at ' + near_plot_pkl)
            plt.close('all')
        else:
            logging.info('Figure exists, skipping ' + near_plot_out)
        
        # (Rep_Joint) Plot representative points x and y as a joint plot for each class
        rep_plot_files = []
        rep_plot_data = {}
        for cls in gdc_cls:
            rep_cls = gdc[gdc[info['target']] == cls]
            rep_plot_data[cls] = rep_cls
            if len(rep_cls) > 1:
                rep_plot_out = os.path.join(xy_plots_subfolder, alias + '_' + cls + '_repxy_joint') + plot['plot_ext']
                rep_plot_pkl = os.path.join(memory_subfolder, alias + '_' + cls + '_repxy_joint.pkl')
                rep_plot_files.append(rep_plot_pkl)
                if not os.path.isfile(rep_plot_out) or not analysis['persist']:
                    fig = plt.figure(figsize=(4.75, 5.5))
                    rep_plot = sns.jointplot(x=rep_cls['repx'], y=rep_cls['repy'], color=plot['plot_color'], marker='.', stat_func=None)
                    rep_plot.ax_joint.set_title(data_title + ': Representative Points Distribution (' + cls + ')')
                    rep_plot.set_axis_labels(xlabel='Representative Coordinate X', ylabel='Representative Coordinate Y')
                    rep_plot.savefig(rep_plot_out, dpi=plot['plot_dpi'])
                    logging.info('Figure saved at ' + rep_plot_out)
                    with open(rep_plot_pkl, 'wb') as f:
                        pickle.dump(rep_plot, f)
                    logging.info('Pickled figure at ' + rep_plot_pkl)
                    plt.close('all')
                else:
                    logging.info('Figure exists, skipping ' + rep_plot_out)
        
        # (Corr_Remove_Bar) Plot removed correlated variables per kept variable
        ocorr_plot_out = os.path.join(plots_subfolder, alias + '_multicorr_reduction_bar') + plot['plot_ext']
        ocorr_plot_pkl = os.path.join(memory_subfolder, alias + '_multicorr_reduction_bar.pkl')
        if not os.path.isfile(ocorr_plot_out) or not analysis['persist']:
            fig = plt.figure(figsize=(8.5, 11))
            ocorr_plot = sns.barplot(x=ocorr['len'].values, y=ocorr['keep'], color=plot['plot_color'])
            ocorr_plot.set_title(data_title + ': Kept Variables from Ordered Multicollinearity Reduction (In Order)')
            ocorr_plot.set_xlabel('Number of Removed Correlated Variables (< ' + str(analysis['high_correlations'][0]) + ', > ' + str(analysis['high_correlations'][1]) + ')')
            ocorr_plot.set_ylabel('Kept Variables')
            ocorr_plot.figure.savefig(ocorr_plot_out, dpi=plot['plot_dpi'])
            logging.info('Figure saved at ' + ocorr_plot_out)
            with open(ocorr_plot_pkl, 'wb') as f:
                pickle.dump(ocorr_plot, f)
            logging.info('Pickled figure at ' + ocorr_plot_pkl)
            plt.close('all')
        else:
            logging.info('Figure exists, skipping ' + ocorr_plot_out)
            
        # (Scores_Line) Plot CV scores per test
        cv_plot_out = os.path.join(plots_subfolder, alias + '_f1scores_line') + plot['plot_ext']
        cv_plot_pkl = os.path.join(memory_subfolder, alias + '_f1scores_line.pkl')
        cv_plot_data = cv_scores
        cv_plot_data['mean_f1_score'] = cv_plot_data['mean_f1_score'].round(2)
        if not os.path.isfile(cv_plot_out) or not analysis['persist']:
            fig = plt.figure(figsize=(8.5, 9.25))
            cv_plot = sns.pointplot(data=cv_plot_data, x='cv_folds', y='mean_f1_score', color=plot['plot_color'])
            cv_plot.set_title(data_title + ': F1 Scores for Cross Validation Tests of Parameter Optimized Random Forest')
            cv_plot.set_ylabel('Mean F1 Score')
            cv_plot.set_xlabel('Cross Validation Folds')
            cv_plot.figure.savefig(cv_plot_out, dpi=plot['plot_dpi'])
            logging.info('Figure saved at ' + cv_plot_out)
            with open(cv_plot_pkl, 'wb') as f:
                pickle.dump(cv_plot, f)
            logging.info('Pickled figure at ' + cv_plot_pkl)
            plt.close('all')
        else:
            logging.info('Figure exists, skipping ' + cv_plot_out)
            
        # (Prob_Bar) Plot mean probability per class
        prob_plot_out = os.path.join(plots_subfolder, alias + '_mean_prob_bar') + plot['plot_ext']
        prob_plot_pkl = os.path.join(memory_subfolder, alias + '_mean_prob_bar.pkl')
        if not os.path.isfile(prob_plot_out) or not analysis['persist']:
            fig = plt.figure(figsize=(8.5, 11))
            prob_plot = sns.barplot(data=rf_mprob, x='max_prob', y='predict', orient='h', color=plot['plot_color'])
            prob_plot.set_title(data_title + ': Mean Prediction Probabilities of Classes for Parameter Optimized Random Forest')
            prob_plot.set_xlabel('Mean Predicted Probability')
            prob_plot.set_ylabel('Predicted Class')
            prob_plot.figure.savefig(prob_plot_out, dpi=plot['plot_dpi'])
            logging.info('Figure saved at ' + prob_plot_out)
            with open(prob_plot_pkl, 'wb') as f:
                pickle.dump(prob_plot, f)
            logging.info('Pickled figure at ' + prob_plot_pkl)
            plt.close('all')
        else:
            logging.info('Figure exists, skipping ' + prob_plot_out)
        
        # (Outlier_Box) Plot outlier measures of classes
        outlier_plot_out = os.path.join(plots_subfolder, alias + '_outlier_classes_box') + plot['plot_ext']
        outlier_plot_pkl = os.path.join(memory_subfolder, alias + '_outlier_classes_box.pkl')
        if not os.path.isfile(outlier_plot_out) or not analysis['persist']:
            fig = plt.figure(figsize=(8.5, 11))
            outlier_plot = sns.boxplot(data=rf_outliers, x='outlier_measure', y='class', orient='h', color=plot['plot_color'], fliersize=3, width=0.3, linewidth=0.5)
            outlier_plot.set_title(data_title + ': Distribution of Outlier Measures from Parameter Optimized Random Forest')
            outlier_plot.set_xlabel('Outlier Measure')
            outlier_plot.set_ylabel('Class')
            outlier_plot.figure.savefig(outlier_plot_out, dpi=plot['plot_dpi'])
            logging.info('Figure saved at ' + outlier_plot_out)
            with open(outlier_plot_pkl, 'wb') as f:
                pickle.dump(outlier_plot, f)
            logging.info('Pickled figure at ' + outlier_plot_pkl)
            plt.close('all')
        else:
            logging.info('Figure exists, skipping ' + outlier_plot_out)
        
        # (Imp_Bar) Plot variable importances
        imp_plot_out = os.path.join(plots_subfolder, alias + '_var_importances_bar') + plot['plot_ext']
        imp_plot_pkl = os.path.join(memory_subfolder, alias + '_var_importances_bar.pkl')
        if not os.path.isfile(imp_plot_out) or not analysis['persist']:
            fig = plt.figure(figsize=(8.5, 11))
            imp_plot = sns.barplot(data=rf_imp, x='importance', y='variable', orient='h', color=plot['plot_color'])
            imp_plot.set_title(data_title + ': Variable Importances of Parameter Optimized Random Forest')
            imp_plot.set_xlabel('Importance')
            imp_plot.set_ylabel('Variable')
            imp_plot.figure.savefig(imp_plot_out, dpi=plot['plot_dpi'])
            logging.info('Figure saved at ' + imp_plot_out)
            with open(imp_plot_pkl, 'wb') as f:
                pickle.dump(imp_plot, f)
            logging.info('Pickled figure at ' + imp_plot_pkl)
            plt.close('all')
        else:
            logging.info('Figure exists, skipping ' + imp_plot_out)
        
        # (Outlier_FC_Bars) Plot variable contributions for outlier classes
        contrib_plot_data = rf_contrib.groupby('class').median()
        contrib_plot_files = []
        for cls in contrib_plot_data.index:
            contrib_plot_out = os.path.join(outlier_plots_subfolder, alias + '_' + cls + '_outlierclass_contrib_bar') + plot['plot_ext']
            contrib_plot_pkl = os.path.join(memory_subfolder, alias + '_' + cls + '_outlierclass_contrib_bar.pkl')
            contrib_plot_files.append(contrib_plot_pkl)
            contrib_cls = contrib_plot_data.loc[cls]
            contrib_cls = contrib_cls.sort_values(ascending=False)
            if not os.path.isfile(contrib_plot_out) or not analysis['persist']:
                fig = plt.figure(figsize=(8.5, 11))
                contrib_plot = sns.barplot(x=contrib_cls.values, y=contrib_cls.index.values, orient='h', color=plot['plot_color'])
                contrib_plot.set_title(data_title + ': Variable Contributions of ' + cls.capitalize() + ' Outliers from Parameter Optimized Random Forest')
                contrib_plot.set_xlabel('Contributions for ' + cls)
                contrib_plot.set_ylabel('Variables of ' + cls)
                contrib_plot.figure.savefig(contrib_plot_out + plot['plot_ext'], dpi=plot['plot_dpi'])
                logging.info('Figure saved at ' + contrib_plot_out)
                with open(contrib_plot_pkl, 'wb') as f:
                    pickle.dump(contrib_plot, f)
                logging.info('Pickled figure at ' + contrib_plot_pkl)
                plt.close('all')
            else:
                logging.info('Figure exists, skipping ' + contrib_plot_out)
                    
        # (Report) Generate a report
        # ---------------------------------------------------------------
        report_template = Template(workflow._report_template)
        
        # (Report_Tables) Obtain tables needed for report
        pd.set_option('display.max_colwidth', -1)
        report_ocorr = ocorr.drop(['len', 'order'], axis=1)
        report_ocorr.columns = ['Kept Variables', 'Removed Variables']
        report_rf_grid = rf_grid.drop(['pkl'], axis=1)
        report_rf_grid.columns = [c.capitalize() for c in report_rf_grid.columns]
        report_tables = {'ocorr_table': report_ocorr,
                         'grid_table': report_rf_grid}
        for k, df in report_tables.items():
            report_tables[k] = df.to_html(index=False, justify='right').replace('border="1"', '').replace(' style="text-align: right;"', '').replace('  class="dataframe"', '')
        pd.set_option('display.max_colwidth', 50)
        
        # (Report_Plots) Obtain plots needed for report
        report_plots = {'cd_plot': cd_plot_pkl,
                        'ocorr_plot': ocorr_plot_pkl,
                        'cv_plot': cv_plot_pkl,
                        'prob_plot': prob_plot_pkl,
                        'imp_plot': imp_plot_pkl,
                        'outlier_plot': outlier_plot_pkl}
        for k, pkl in report_plots.items():
            with open(pkl, 'rb') as f:
                kplot = pickle.load(f)
                report_plots[k] = workflow.gen_html_plot(kplot.figure, dpi=300, format='png')
            plt.close('all')
        
        # (Generate_Report) Generate the html report
        report_out = os.path.join(project_workspace, alias + '_report.html')
        report_misc = {'title': alias, 'exp_title': info['title']}
        report_vars = {**report_misc, **report_plots, **report_tables}
        report_html = report_template.render(report_vars)
        with open(report_out, 'w') as f:
            f.write(report_html)
            
            