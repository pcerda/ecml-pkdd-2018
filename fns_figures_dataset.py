import numpy as np
import pandas as pd
import glob
import os
import sys

import datetime
from joblib import Parallel, delayed
import string
from itertools import accumulate
from scipy import interpolate
import seaborn as sns
import matplotlib.pyplot as plt

import ngram
import time
import Levenshtein as lev

from sklearn import random_projection
from sklearn import svm
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import metrics
from sklearn import neural_network
from sklearn import ensemble


CE_HOME = os.environ.get('CE_HOME')
sys.path.append(os.path.abspath(os.path.join(
    CE_HOME, 'python', 'categorical_encoding')))
from datasets import Data
from scipy import stats
from fns_categorical_encoding import *


def score_plot(datasets, conditions, condition, score, percentile_thresh=1,
               delta_text=0,
               delta_top=0, percentile_dict={'levenshtein-ratio': -1,
                                             'jaro-winkler': -1,
                                             '3-gram': -1}):
    plt.close("all")
    fig, ax = plt.subplots()

    # # add similarity distribution ###########################################
    # scale = .035
    # bins = np.linspace(0, 1, num=11)
    # for i, dataset in enumerate(sorted(datasets)):
    #     data = Data(dataset).get_df()
    #     sm_cols = [col for col in data.col_action
    #                if data.col_action[col] == 'se']
    #     print(dataset)
    #     distances = []
    #     for sm_col in sm_cols[:1]:
    #         print('Column name: %s' % sm_col)
    #         A = data.df[sm_col][:10000].astype(str)
    #         B = data.df[sm_col].unique().astype(str)
    #         sm = similarity_matrix(A, B, conditions['Distance'], -1)
    #         print(sm.shape)
    #         # take the 10% highest distances for each value
    #         sm_nmax = np.array([sorted(row)[:-1]
    #                             for row in sm])
    #         distances += list(sm_nmax.ravel())
    #     bin_counts = [0 for bin in bins]
    #     bin_width = 1/(len(bins)-1)
    #     distances2 = np.zeros(len(distances))
    #     for i, distance in enumerate(distances):
    #         bin_number = int(distance // bin_width)
    #         bin_counts[bin_number] += 1
    #         distances2[i] = bin_number * bin_width
    #     bin_counts = np.array(bin_counts)/len(distances)
    #     s = interpolate.interp1d(bins, bin_counts)
    #     kernel = stats.gaussian_kde(distances2, bw_method=.6)
    #     x = np.linspace(0, 1, 201)
    #     plt.semilogy(x, s(x)*scale-.02, '--')
    # #########################################################################

    df_all = pd.DataFrame()
    for dataset in sorted(datasets):
        data = Data(dataset)
        results_path = os.path.join(data.path, 'output', 'results')
        figures_path = os.path.join(data.path, 'output', 'figures')
        create_folder(data.path, 'output/figures')

        files = glob.glob(os.path.join(results_path, '*'))
        files, params = file_meet_conditions(dataset, files, conditions)
        print('Relevant files:')
        for f in files:
            print(f.split('..')[-1])
            df = pd.read_csv(f)
            df = df.drop_duplicates(subset=df.columns[1:])
            df_ohe = df[df.threshold == 1.0].set_index('fold')[['score']]
            df_ohe.rename(columns={'score': 'score(ohe)'}, inplace=True)
            df = df.join(df_ohe, on='fold')
            df['score-score(ohe)'] = df['score'] - df['score(ohe)']
            df['Dataset'] = data.name
            df['Classifier'] = results_parameters(f)['Classifier'][:-4]
            df['Distance'] = results_parameters(f)['Distance']
            df['TyposProb'] = results_parameters(f)['TyposProb']
            percentiles = percentile_dict[results_parameters(f)['Distance']]
            percentiles[10] = 100
            if percentile_thresh == 1:
                for i in range(len(df)):
                    df.loc[i, 'threshold'] = percentiles[int(df.loc[i, 'threshold']*10)]

            name = f.split('/')[-1]
            name = name.split('_')
            dict_name = {}
            for n in name:
                key, value = [n.split('-')[0], '-'.join(n.split('-')[1:])]
                dict_name[key] = value
            df_all = pd.concat([df_all, df], axis=0)
            if percentile_thresh == 1:
                df_all = df_all.drop_duplicates(subset=['threshold', 'Distance', 'fold'])
    # plot scores
    values = df_all[condition].unique()
    sns.set_palette(set_colors(values, condition))

    sns.tsplot(data=df_all, time='threshold', unit='fold',
               condition=condition,
               value=score, ci=95,
               ax=ax, marker='.',
               markersize=10)

    max_all = df_all[score].max()
    min_all = df_all[score].min()
    if min_all <= 0:
        ax.axhline(y=0, xmin=-10, xmax=110, linewidth=1, color='grey')
    sns.plt.ylim([min_all, max_all])
    sns.despine(bottom=True, right=False, trim=True)
    sns.despine()
    sns.plt.xlim([-10, 110])
    sns.plt.ylim([min_all-(max_all-min_all)*.1,
                  max_all+(max_all-min_all)*.1 + delta_top])
    df.groupby('threshold')

    mean_score_ohe = np.mean(df[score][df.threshold == 1])
    ax.text(0, min_all+delta_text, 'Raw\nsimilarity\nencoding', fontsize=14,
            horizontalalignment='center', verticalalignment='top',
            color='gray')
    ax.text(100, min_all+delta_text, 'One-hot\nencoding',
            fontsize=14, color='gray',
            horizontalalignment='center',
            verticalalignment='top')
    ax.set_xlabel('Hard-thresholding value', fontsize=16)
    if score == 'score':
        ax.set_ylabel('Score', fontsize=16)
    elif score == 'score-score(ohe)':
        ax.set_ylabel('Score - Score(one-hot-encoding)', fontsize=16)

    ax.tick_params(axis='both', which='major', labelsize=14)

    leg = ax.get_legend()
    leg = ax.legend(fontsize=14, ncol=1)
    leg.set_title(condition, prop={'size': 16})

    # sns.axes_style()
    # sns._orig_rc_params
    return ax


def distances_dist(datasets, conditions, scale=1):
    plt.close("all")
    fig, ax = plt.subplots()
    bins = np.linspace(0, 1, num=11)
    for i, dataset in enumerate(sorted(datasets)):
        data = Data(dataset).get_df()
        sm_cols = [col for col in data.col_action
                   if data.col_action[col] == 'se']
        print(dataset)
        distances = []
        for sm_col in sm_cols[:1]:
            print('Column name: %s' % sm_col)
            A = data.df[sm_col][:10000].astype(str)
            B = data.df[sm_col].unique().astype(str)
            sm = similarity_matrix(A, B, conditions['Distance'], -1)
    #         print(sm.shape)
    #         # take the 10% highest distances for each value
    #         sm_nmax = np.array([sorted(row)[:-1]
    #                             for row in sm])
    #         distances += list(sm_nmax.ravel())
    #     bin_counts = [0 for bin in bins]
    #     bin_width = 1/(len(bins)-1)
    #     distances2 = np.zeros(len(distances))
    #     for i, distance in enumerate(distances):
    #         bin_number = int(distance // bin_width)
    #         bin_counts[bin_number] += 1
    #         distances2[i] = bin_number * bin_width
    #     bin_counts = np.array(bin_counts) #/len(distances)
    #     s = interpolate.interp1d(bins, bin_counts)
    #     kernel = stats.gaussian_kde(distances2, bw_method=.6)
        x = np.linspace(0, 1, 11)
        # y = list(reversed(list(accumulate(list(reversed(bin_counts))))))
        # plt.semilogy(x, s(x)*scale, label=dataset)
        plt.semilogy(x, ball_elements(sm, bins)/sm.shape[0],
                     label=dataset)
    plt.legend(fontsize=14)
    sns.plt.xlim([0, 1])
    sns.plt.ylim([1, 2000])
    # sns.despine(bottom=False, left=False, right=True, trim=True)
    # plt.yticks([], [])
    # y_ticks = np.array([val/10 for val in ax.get_yticks()])
    # ax.set_yticklabels(y_ticks)
    ax.set_xlabel('Similarity', fontsize=16)
    ax.tick_params(axis='x', which='major', labelsize=14)
    filename = 'DistanceDist_' + '_'.join([key + '-' + conditions[key]
                                           for key in conditions]) + '.pdf'
    plt.savefig(os.path.join(os.getcwd(), '..', 'figures', filename),
                transparent=False, bbox_inches='tight', pad_inches=0.2)


def word_freq(datasets, conditions):
    plt.close("all")
    fig, ax = plt.subplots()
    values = sorted(datasets)
    sns.set_palette(set_colors(values, 'Dataset'))
    for i, dataset in enumerate(sorted(datasets)):

        data = Data(dataset).get_df()
        sm_cols = [col for col in data.col_action
                   if data.col_action[col] == 'se']

        for sm_col in sm_cols[:1]:
            counts = data.df[sm_col].value_counts()

            # Plot histogram using matplotlib bar().
            indexes = list(counts.index)
            vals = counts.values
            f = interpolate.interp1d(np.linspace(0, 1, len(indexes)), vals)
            x = np.linspace(0, 1, 1000)
            plt.semilogy(x, f(x), label=dataset, linewidth=3.0)
    plt.legend(fontsize=14)
    sns.plt.ylim([1, ax.get_ylim()[1]])
    sns.despine(bottom=True, right=False, trim=True)
    sns.despine()
    sns.plt.xlim([-.03, 1.03])
    sns.plt.ylim([pow(10, -.2), ax.get_ylim()[1]])
    plt.xticks([0, 1], ['', ''])
    plt.minorticks_off()
    ax.set_xlabel('Classes', fontsize=16)
    ax.set_ylabel('log(Frequency)', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    filename = ('ClassFreq_' + '_'.join([key + '-' + conditions[key]
                                         for key in conditions]) + '.pdf')
    plt.savefig(os.path.join(os.getcwd(), '..', 'figures', filename),
                transparent=False, bbox_inches='tight', pad_inches=0.2)


def ball_elements(SE, bins):
    output = np.zeros(len(bins))
    for i, bin_ in enumerate(bins):
        for row in SE:
            output[i] += len(row[row >= bin_])
    return output


def number_elements_in_ball_by_row(A, radiuses):
    outputs = []
    radiuses = sorted(radiuses)
    for i, row in enumerate(A):
        if i/1000.0 == int(i/1000.0):
            print(i)
        row = sorted(row)
        j = 0
        output_row = []
        for radius in radiuses:
            while row[j] <= radius:
                j += 1
            output_row.append(j)
        outputs.append(output_row)
    return outputs


def set_list(condition, clf_type='None'):
    if clf_type == 'regression':
        if condition == 'Classifier':
            cnd_dict = {'Ridge': 0,
                        'GradientBoosting': 2,
                        'RandomForest': 3,
                        'MLP': 5,
                        'KNeighbors': 4}
    else:
        if condition == 'Classifier':
            cnd_dict = {'Ridge': 0,
                        'LogisticRegression': 1,
                        'GradientBoosting': 2,
                        'RandomForest': 3,
                        'MLP': 5,
                        'KNeighbors': 4}

    if condition == 'Dataset':
        cnd_dict = {'docs_payments': 0,
                    'midwest_survey': 1,
                    'medical_charge': 2,
                    'adult': 3,
                    'beer_reviews': 4,
                    'road_safety': 5,
                    'employee_salaries': 6,
                    'indultos_espana': 7}
    if condition == 'Distance':
        cnd_dict = {'one-hot': 4,
                    'levenshtein-ratio': 0,
                    '3-gram': 1,
                    'sorensen': 5,
                    'jaccard': 2,
                    'jaro-winkler': 3}
    if condition == 'TyposProb':
        cnd_dict = {'0.00': 0,
                    '0.01': 1,
                    '0.02': 2,
                    '0.05': 3,
                    '0.10': 4,
                    '0.20': 5}

    return cnd_dict
