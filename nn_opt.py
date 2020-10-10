#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 07:47:14 2020

@author: brandon
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import resample
import scipy.sparse
from sklearn.metrics import balanced_accuracy_score, recall_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

import time

import joblib

import mlrose_hiive as mlrose

RANDOM_STATE = 0
SIZE_LIMIT = 2000
MAX_ITERS = 4000
LINE_WIDTH = 3
RUNS_PER_ALGO = 10
FIG_WIDTH = 4
FIG_HEIGHT = 3
fig_num = 0

# TODO: Write report while generating results.
# TODO: Plot times
# TODO: Improve initial weights

#pd.set_option('display.max_columns', None)

def rebalance(name, X, y, f=np.max):
    bins = np.bincount(y)
    print('bins for %s: %s' % (name, str(bins)))
    class_size = f(bins)
    X_balanced, y_balanced = np.empty((0, X.shape[1])), np.empty((0), dtype=int)
    for i in range(bins.size):
        X_upsampled, y_upsampled = resample(
                X[y == i],
                y[y == i],
                replace=True,
                n_samples=class_size,
                random_state=RANDOM_STATE)
        X_balanced = np.vstack((X_balanced, X_upsampled))
        y_balanced = np.hstack((y_balanced, y_upsampled))

    print('balanced bins for %s: %s' % (name, str(np.bincount(y_balanced))))
    return X_balanced, y_balanced

def limit_size(name, X, y, size):
    if y.size < size:
        return X, y
    X_limited, y_limited = np.empty((0, X.shape[1])), np.empty((0), dtype=int)
    bins = np.bincount(y)
    for i in range(bins.size):
        if bins[i] > size:
            X_downsampled, y_downsampled = resample(
                    X[y == i],
                    y[y == i],
                    replace=True,
                    n_samples=size,
                    random_state=RANDOM_STATE)
        else:
            X_downsampled = X[y == i]
            y_downsampled = y[y == i]
        X_limited = np.vstack((X_limited, X_downsampled))
        y_limited = np.hstack((y_limited, y_downsampled))
    print('limited bins for %s: %s' % (name, str(np.bincount(y_limited))))
    return X_limited, y_limited

def dexter():
    sparse_matrix = scipy.sparse.load_npz('data/dexter/dexter.npz')
    X = sparse_matrix[:, :-1].toarray().astype(np.int16)
    y = sparse_matrix[:, -1].toarray().astype(np.int8).reshape(-1)
    print(y)
    print('Dexter bin counts', np.bincount(y))
    return X, y

def polish_bankruptcy():
    #df = pd.read_csv(base_dir + 'polish_bankruptcy/5year.arff', encoding='utf-8', header=None, na_values=['?'])
    df = pd.read_csv('data/polish_bankruptcy/5year.arff', encoding='utf-8', header=None, na_values=['?'])
    df.fillna(df.mean(), inplace=True)
    print(df.head())
    X_df = df.iloc[:, :-1]
    X = X_df.values

    y_df = df.iloc[:, -1]
    y = y_df.values
    #X, y = rebalance('bank', X, y, f=np.min)

    return X, y

class Problem:
    def __init__(self, fn, name):
        self.name = name
        self.fn = fn
        X, y = fn()
        print('Bin counts', np.bincount(y))
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
        # SMOTE
        X_train, y_train = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train, y_train)
        # Limit size
        X_train, y_train = limit_size(data_set.__name__ + ' training', X_train, y_train, SIZE_LIMIT)
        X_test, y_test = limit_size(data_set.__name__ + ' testing', X_test, y_test, SIZE_LIMIT)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        self.X = X
        self.y = y
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

class NamedGeomDecay(mlrose.GeomDecay):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        mlrose.GeomDecay.__init__(self, **kwargs)
    
    def __repr__(self):
        return str(self.kwargs)

class Algorithm:
    def __init__(self, name, algorithm, param_grid = {}, **kwargs):
        self.name = name
        self.param_grid = param_grid
        self.algorithm = algorithm
        self.default_args = kwargs

    def get_args(self, **kwargs):
        args = {**self.default_args, **kwargs}
        schedule_args = {}
        for k in ['init_temp', 'decay', 'min_temp']:
            if k in args.keys():
                schedule_args[k] = args[k]
                del args[k]
        if schedule_args:
            args['schedule'] = NamedGeomDecay(**schedule_args)
        if 'learning_rate' in args and 'clip_max' in args:
            args['learning_rate'] = min(args['learning_rate'], args['clip_max'])
        return args
    
    def __repr__(self):
        return "%s" % (self.name)
    
    """
    def __getstate__(self):
        return (self.name, self.algorithm.__name__, self.param_grid, self.default_args)
    
    def __setstate__(self, state):
        algorithm = getattr(mlrose, state[1])
        Algorithm.__init__(self, state[0], algorithm, state[2], **state[3])
    """

simulated_annealing_grid = {
        #'schedule': [mlrose.GeomDecay, mlrose.ArithDecay, mlrose.ExpDecay]
        'learning_rate': (np.logspace(start=3, stop=-2, num=20), 'log'),
        'max_attempts': (range(1, 41, 2), 'linear'),
        'init_temp': (np.logspace(-3, 4, 20), 'log'),
        'decay': (1-np.logspace(-0.01, -3.5, 20), 'log'),
        'min_temp': (np.logspace(-1, -7, 20), 'log'),
        'clip_max': (np.logspace(1, 10, 20), 'log'),
        }

rhc_grid = {
        'learning_rate': (np.logspace(start=2, stop=-3, num=20), 'log'),
        'max_attempts': (range(1, 31, 2), 'linear'),
        'restarts': (range(1, 31, 2), 'linear'),
        'clip_max': (np.logspace(1, 10, 20), 'log'),
        }

genetic_grid = {
        'learning_rate': (np.logspace(start=3, stop=-7, num=20), 'log'),
        'pop_size': (range(50, 701, 50), 'linear'),
        #'mutation_prob': (np.logspace(-2, -0.1, 40), 'log'),
        'mutation_prob': (np.linspace(0.1, 0.5, 40), 'linear'),
        'max_attempts': (range(1, 101, 2), 'linear'),
        'clip_max': (np.logspace(-3, 3, 20), 'log'),
        }

gradient_descent_grid = {
        'learning_rate': (np.logspace(start=3, stop=-7, num=20), 'log'),
        'clip_max': (np.logspace(1, 10, 20), 'log'),
        }

ALGORITHMS = [
    Algorithm('GD', 'gradient_descent', gradient_descent_grid,
              learning_rate = 1.4e-4, max_iters=MAX_ITERS, clip_max=1e3),
    Algorithm('RHC', 'random_hill_climb', rhc_grid,
              max_iters=MAX_ITERS, max_attempts=20, restarts=10,
              learning_rate = 2.5, clip_max=1e3),
    Algorithm('SA', 'simulated_annealing', simulated_annealing_grid,
              learning_rate = 2, schedule=mlrose.GeomDecay(), max_attempts=30,
              max_iters=MAX_ITERS, clip_max=1e3, init_temp=1, decay=1-0.01, min_temp=3.36e-5),
    Algorithm('GA', 'genetic_alg', genetic_grid, max_iters=400,
              learning_rate = 0.01, max_attempts=10, clip_max=0.1),
]

def train_nn(X_train, y_train, **kwargs):
    data_hash = joblib.hash([kwargs, X_train, y_train])
    file_name = "cache/nn/%s.dump" % (data_hash)
    if os.path.exists(file_name):
        print("loading nn from cache for hash=%s, args=%s" % (data_hash, kwargs))
        return joblib.load(file_name)
    
    print("Building nn with %s" % (kwargs))
    nn = mlrose.NeuralNetwork(**kwargs)
    start = time.time()
    nn.fit(X_train, y_train)
    nn.time = time.time() - start
    joblib.dump(nn, file_name, compress=3)
    return nn

def run_nn(algorithm, problem, **kwargs):
    args = algorithm.get_args(**kwargs)
    if not 'random_state' in args:
        args['random_state'] = RANDOM_STATE
    nn = train_nn(problem.X_train, problem.y_train, hidden_nodes=[35], activation="tanh",
                  algorithm=algorithm.algorithm, early_stopping=True,
                  curve=True, **args)
    print("For %s" % (algorithm))
    print("loss: %s" % (nn.loss))
    print("fitness curve: %s %s" % (nn.fitness_curve[:5], nn.fitness_curve[-5:]))

    y_predict = nn.predict(problem.X_train)
    #print("probs(train): %s" % (nn.predicted_probs))
    train_score = balanced_accuracy_score(problem.y_train, y_predict)
    train_sensitivity_score = recall_score(problem.y_train, y_predict)
    # https://stackoverflow.com/a/59396145
    train_specificity_score = recall_score(problem.y_train, y_predict, pos_label=0)
    print("Score (train): %s" % (train_score))

    y_predict = nn.predict(problem.X_test)
    #print("probs(test): %s" % (nn.predicted_probs))
    test_score = balanced_accuracy_score(problem.y_test, y_predict)
    print("Score (test): %s" % (test_score))
    print()
    return nn.loss, train_score, train_sensitivity_score, train_specificity_score, test_score, nn.time, nn.fitness_curve

def run_fitness_by_param(algorithm, problem, param_key, param_values):
    x = param_values
    num_points = len(x)
    losses = []
    scores = []
    sensitivity_scores = []
    specificity_scores = []
    num_runs = 3
    for i in range(num_points):
        args = {param_key: x[i]}
        print("%s %s=%s" % (algorithm, param_key, x[i]))
        iter_losses = np.zeros((num_runs))
        iter_train_scores = np.zeros((num_runs))
        iter_scores = np.zeros((num_runs))
        iter_sensitivity_scores = np.zeros((num_runs))
        iter_specificity_scores = np.zeros((num_runs))
        for j in range(num_runs):
            iter_losses[j], iter_train_scores[j], iter_sensitivity_scores[j], iter_specificity_scores[j], \
                iter_scores[j], _, _ = run_nn(algorithm, problem, random_state=RANDOM_STATE + j, **args)
        losses.append(np.median(iter_losses))
        scores.append(np.median(iter_train_scores))
        sensitivity_scores.append(np.median(iter_sensitivity_scores))
        specificity_scores.append(np.median(iter_specificity_scores))
    return x, losses, scores, sensitivity_scores, specificity_scores

def best_param_string(value, scale):
    if isinstance(value, np.float64):
        if scale == 'log':
            return '%.3e' % (value)
        else:
            return '%.3f' % (value)
    return str(value)

def run_fitness_over_param_grid(algorithm, problem):
    global fig_num
    for param_key, (param_values, scale) in algorithm.param_grid.items():
        fig_num += 1
        plt.figure(fig_num)

        print('\tStarting %s' % (param_key))
        x, losses, scores, sensitivity_scores, specificity_scores = run_fitness_by_param(algorithm, problem, param_key, param_values)
        print('\tFinished %s' % (param_key))
        if param_key == 'decay':
            x = 1 - x
            param_key = '1 - decay'

        # https://matplotlib.org/gallery/api/two_scales.html
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        best = x[np.argmax(scores)]
        ax1.set_title('%s - %s (%s)' % (problem.name, algorithm.name, param_key))
        ax1.set_xlabel('%s' % (param_key))
        ax1.set_ylabel('Balanced Accuracy (best=%s)' % (best_param_string(best, scale)), color=color)
        ax1.plot(x, scores, linewidth=LINE_WIDTH, markersize=LINE_WIDTH, color=color, label='Balanced Accuracy')
        ax1.plot(x, sensitivity_scores, linewidth=1, markersize=1, linestyle="--", color=color, label='Sensitivity')
        ax1.plot(x, specificity_scores, linewidth=1, markersize=1, linestyle=":", color=color, label='Specificity')
        #ax1.fill_between(x, means + stds, means - stds, alpha=0.15, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend()
        best_value = algorithm.default_args[param_key]
        ax1.axvline(x=best_value, linestyle='--', linewidth=1, color='black')
        ax1.set_ylim([0, 1.03])

        ax2 = ax1.twinx()
        color = 'tab:blue';
        best = x[np.argmin(losses)]
        ax2.set_ylabel('Log Loss (best=%s)' % (best_param_string(best, scale)), color=color)
        ax2.plot(x, -np.array(losses), linewidth=LINE_WIDTH, markersize=LINE_WIDTH, color=color)
        #ax2.fill_between(x, fn_means + fn_stds, fn_means - fn_stds, alpha=0.15, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim([min(-2.0, -max(losses) * 1.03), 0])

        #plt.grid()
        plt.xscale(scale)
        plt.gcf().set_size_inches(FIG_WIDTH, FIG_HEIGHT)
        plt.savefig('tuning_plots/nn/%s_%s_%s.png' % (problem.name, algorithm.name, param_key), bbox_inches='tight')
        plt.close()

def plot_by_iteration(fig, problem, algorithm):
    curves = []
    train_scores = np.zeros((RUNS_PER_ALGO))
    test_scores = np.zeros((RUNS_PER_ALGO))
    timings = np.zeros((RUNS_PER_ALGO))
    for i in range(RUNS_PER_ALGO):
        loss, train_score, _1, _2, test_score, timing, curve = run_nn(algorithm, problem, random_state=RANDOM_STATE + i)
        curves.append(curve)
        train_scores[i] = train_score
        test_scores[i] = test_score
        timings[i] = timing
    max_len = max([len(curve) for curve in curves])
    results = np.zeros((RUNS_PER_ALGO, max_len))
    multiplier = 1
    if algorithm.algorithm == 'gradient_descent':
        multiplier = -1
    for i in range(RUNS_PER_ALGO):
        results[i, :len(curves[i])] = multiplier * np.array(curves[i])
    means = np.mean(results, axis=0)
    stds = np.std(results, axis=0)
    x = np.arange(max_len)
    plt.figure(fig.number)
    plt.plot(x, means, linewidth=LINE_WIDTH, markersize=LINE_WIDTH, label=algorithm.name)
    plt.fill_between(x, means + stds, means - stds, alpha=0.15)
    plt.xlabel('Number of iterations')
    plt.ylabel('Log loss')
    return train_scores, test_scores, timings

def label_bars(text_format, **kwargs):
    ax = plt.gca()
    fig = plt.gcf()
    bars = ax.patches

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for bar in bars:
        text = text_format % (bar.get_width())
        color = 'white'
        va = 'center'
        ha = 'center'
        text_x = bar.get_x() + bar.get_width() / 2
        text_y = bar.get_y() + bar.get_height() / 2
        if bar.get_window_extent(renderer).width < 10:
            color = 'black'
            ha = 'left'
            text_y = bar.get_y() + 1.5 * bar.get_width()

        ax.text(text_x, text_y, text, ha=ha, va=va, color=color, **kwargs)

# https://matplotlib.org/3.1.1/gallery/units/bar_unit_demo.html
def comparison_bar_chart(data_set_name, train, train_err, test, test_err, text_format='%.2f', **kwargs):
    fig, ax = plt.subplots()
    ind = np.arange(len(train))
    width = 0.4
    ax.barh(ind, train, width, label='Train', yerr=train_err)
    ax.barh(ind + width, test, width, label='Test', yerr=test_err)
    ax.set_xlabel('Balanced Accuracy')
    ax.set_title('Balanced Accuracy for %s' % (data_set_name))
    ax.set_yticks(ind + width / 2)
    algorithm_names = [algorithm.name for algorithm in ALGORITHMS]
    ax.set_yticklabels(algorithm_names)

    ax.legend(loc='upper left')
    ax.autoscale_view()
    ax.invert_yaxis()
    label_bars('%.2f')

    # Based on https://stackoverflow.com/a/50354131
    fig = plt.gcf()
    bars = ax.patches
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for bar in bars:
        text = text_format % (bar.get_width())
        color = 'white'
        va = 'center'
        ha = 'center'
        text_x = bar.get_x() + bar.get_width() / 2
        text_y = bar.get_y() + bar.get_height() / 2
        if bar.get_window_extent(renderer).width < 10:
            color = 'black'
            ha = 'left'
            text_y = bar.get_y() + 1.5 * bar.get_width()

        ax.text(text_x, text_y, text, ha=ha, va=va, color=color, **kwargs)

# https://matplotlib.org/examples/lines_bars_and_markers/barh_demo.html
def simple_bar_chart(labels, data, err, text_format='%1.2f', **kwargs):
    ax = plt.gca()
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, data, xerr=err)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    label_bars(text_format)

if __name__ == "__main__":
    if not os.path.exists('cache/nn'):
        os.makedirs('cache/nn')
    if not os.path.exists('plots/nn'):
        os.makedirs('plots/nn')
    if not os.path.exists('tuning_plots/nn'):
        os.makedirs('tuning_plots/nn')
        
    data_set = polish_bankruptcy
    problem = Problem(polish_bankruptcy, 'Polish Bankruptcy')

    fig_num += 1
    iter_fig = plt.figure(fig_num)
    plt.title('Loss by iteration for %s' % (problem.name))

    train_scores = np.zeros((len(ALGORITHMS), RUNS_PER_ALGO))
    test_scores = np.zeros((len(ALGORITHMS), RUNS_PER_ALGO))
    timings = np.zeros((len(ALGORITHMS), RUNS_PER_ALGO))
    for i in range(len(ALGORITHMS)):
        algorithm = ALGORITHMS[i]
        curr_train_scores, curr_test_scores, curr_timings = plot_by_iteration(iter_fig, problem, algorithm)
        train_scores[i] = curr_train_scores
        test_scores[i] = curr_test_scores
        timings[i] = curr_timings

    fig_num += 1
    plt.figure(fig_num)
    algorithm_names = [algorithm.name for algorithm in ALGORITHMS]
    simple_bar_chart(algorithm_names, np.mean(timings, axis=1), np.std(timings, axis=1))
    plt.title('%s convergence time' % (problem.name))
    #plt.ylabel('Algorithm')
    plt.xlabel('Time (s)')
    plt.gcf().set_size_inches(FIG_WIDTH, FIG_HEIGHT)
    plt.savefig('plots/nn/%s_%s.png' % (problem.name, 'convergence_times'), bbox_inches='tight')
    plt.close()

    fig_num += 1
    plt.figure(fig_num)
    y_train, y_train_err = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
    y_test, y_test_err = np.mean(test_scores, axis=1), np.std(test_scores, axis=1)
    comparison_bar_chart(problem.name, y_train, y_train_err, y_test, y_test_err)
    plt.title('Balanced Accuracy by model for %s' % (problem.name))
    plt.gcf().set_size_inches(FIG_WIDTH, FIG_HEIGHT)
    plt.savefig('plots/nn/%s_bar.png' % (problem.name), bbox_inches='tight')
    plt.close()

    plt.figure(iter_fig.number)
    plt.grid()
    plt.legend()
    plt.gcf().set_size_inches(FIG_WIDTH, FIG_HEIGHT)
    plt.savefig('plots/nn/%s_iterations.png' % (problem.name), bbox_inches='tight')
    plt.close()

    for algorithm in ALGORITHMS:
        run_fitness_over_param_grid(algorithm, problem)