#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 06:48:47 2020

@author: brandon
"""

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, dump, load

import os

# TODO: Improve fn_call counts

RUNS_PER_ALGO = 32
DEFAULT_LENGTH = 50
LENGTH_VALUES = [i for i in range(10, DEFAULT_LENGTH+1, 10)]
DO_PARALLEL = True
LINE_WIDTH = 3
fig_num = 0
FIG_WIDTH = 4
FIG_HEIGHT = 3


from run_job import run_iteration, ALGORITHMS, FITNESS_FNS, instance, MAX_ITERS, RANDOM_STATE_OFFSET


def throw_if_error(results):
    # From https://stackoverflow.com/a/26096355
    for result in results:
        if isinstance(result, str):
            result.re_raise()

def my_parallel():
    return Parallel(n_jobs=-1)

def run_length_iteration(problem, length, algorithm, **kwargs):
    file_name = "cache/%s-%s-%s-%s-%s-withtime.dump" % (problem.name, length,
                                               algorithm.name,
                                               algorithm.default_args, kwargs)
    if os.path.exists(file_name):
        print("loading %s from cache" % (file_name))
        return load(file_name)
    if DO_PARALLEL:
        # https://medium.com/@mjschillawski/quick-and-easy-parallelization-in-python-32cb9027e490
        results = my_parallel()(
                delayed(run_iteration)(i, problem.name, length, algorithm.name, **kwargs)
                for i in range(RUNS_PER_ALGO))
    else:
        results = []
        for i in range(RUNS_PER_ALGO):
            results.append(run_iteration(0, problem.name, length, algorithm.name, **kwargs))

    throw_if_error(results)

    #normalization = results[0][4]
    #fitnesses = np.array([result[0] for result in results])
    fitnesses = np.array([result[0] for result in results])
    iters = np.array([result[1] for result in results])
    fn_calls = np.array([result[3] for result in results])
    results = np.array([result[2] for result in results])
    timings = np.array([result[4] for result in results])
    return_value = fitnesses, fn_calls, results, iters, timings
    print("Saving %s" % (file_name))
    dump(return_value, file_name, compress=3)
    return return_value

#cached_run_length_iteration = mem.cache(run_length_iteration)

def run_length_iterations(delayed_calls):
    results = Parallel(n_jobs=1)(delayed_calls)
    y = [result[0] for result in results]
    fn_calls = [result[1] for result in results]
    iters = [result[3] for result in results]
    return y, fn_calls, iters

def run_fitness_by_length(problem, algorithm, **kwargs):
    #print("Running fitness by length for %s-%s" % (problem.name, algorithm.name))
    x = LENGTH_VALUES
    num_points = len(x)
    #delayed_calls = []
    y, fn_calls, iters, timings = [], [], [], []
    for i in range(num_points):
        length = x[i]
        #print("\tlength=%d" % (length))
        curr_y, curr_fn_calls, _, curr_iters, curr_timings = run_length_iteration(problem, length, algorithm, **kwargs)
        y.append(curr_y)
        fn_calls.append(curr_fn_calls)
        iters.append(curr_iters)
        timings.append(curr_timings)

        #delayed_calls.append(delayed(run_length_iteration)(problem, length, algorithm, **kwargs))
    #y, fn_calls, iters = run_length_iterations(delayed_calls)
    return x, y, fn_calls, timings

def run_fitness_by_param(problem, length, algorithm, param_key, param_values):
    x = param_values
    num_points = len(x)
    #delayed_calls = []
    y, fn_calls, iters = [], [], []
    for i in range(num_points):
        args = {param_key: x[i]}
        print("\t%s=%s" % (param_key, x[i]))
        curr_y, curr_fn_calls, _, curr_iters, timings = run_length_iteration(problem, length, algorithm, **args)
        y.append(curr_y)
        fn_calls.append(curr_fn_calls)
        iters.append(curr_iters)
        #delayed_calls.append(delayed(run_length_iteration)(problem, length, algorithm, **args))
    #y, fn_calls, iters = run_length_iterations(delayed_calls)

    return x, y, fn_calls

def run_fitness_by_iteration(problem, length, algorithm, **kwargs):
    global problem_figures
    results = np.zeros((RUNS_PER_ALGO, MAX_ITERS))

    fitnesses, fn_calls, results, iters, timings = run_length_iteration(problem, length,
                                                                        algorithm, **kwargs)
    max_iters = max(iters)
    fn_calls_per_iter = fn_calls / iters
    x_fn = np.zeros((RUNS_PER_ALGO, max_iters))
    x_iter = np.zeros((RUNS_PER_ALGO, max_iters))
    for i in range(RUNS_PER_ALGO):
        x_iter[i] = np.arange(max_iters)
        x_fn[i] = np.arange(max_iters) * fn_calls_per_iter[i]
    results = results[:, :max_iters]
    #y, y_err = np.mean(results, axis=0), np.std(results, axis=0)

    return x_fn, x_iter, results


def run_fitness_over_param_grid(length, algorithm):
    global fig_num
    for param_key, (param_values, scale) in algorithm.param_grid.items():
        param_label = param_key
        if param_key == 'decay':
            param_label = '1 - decay'

        fig_num += 1
        param_fig, param_ax1 = plt.subplots()
        param_ax1.set_title('%s length=%s (%s)' % (algorithm.name, length, param_label))
        best_value = ""
        # Hack to avoid having to re-run everything
        if param_key == 'mutation_prob':
            best_value = 0.1
        else:
            best_value = algorithm.default_args[param_key]
        param_ax1.set_xlabel('%s (best=%s)' % (param_label, best_value))
        param_ax1.set_ylabel('Avg Fitness % +/- Std Dev (solid line)')
        param_ax2 = param_ax1.twinx()
        param_ax2.set_ylabel('Count of Function Calls (dashed line)')

        for problem in FITNESS_FNS:
            print('\tStarting %s' % (param_label))
            x, y, fn_calls = run_fitness_by_param(problem, length, algorithm, param_key, param_values)
            print('\tFinished %s' % (param_label))
            if param_key == 'decay':
                x = 1 - x
            means = np.mean(y, axis=1)
            stds = np.std(y, axis=1)
            fn_means = np.mean(fn_calls, axis=1)
            fn_stds = np.std(fn_calls, axis=1)

            #plt.figure(fig_num)
            _, normalization = instance(problem, length)
            p = param_ax1.plot(x, means / normalization, linewidth=LINE_WIDTH, markersize=LINE_WIDTH, label=problem.name)
            param_ax2.plot(x, fn_means, linestyle="--", linewidth=1, markersize=LINE_WIDTH, color=p[-1].get_color())

            # https://matplotlib.org/gallery/api/two_scales.html
            fig_num += 1
            fig, ax1 = plt.subplots()
            color = 'tab:red'
            ax1.set_title('%s - %s length=%s (%s)' % (problem.name, algorithm.name, length, param_label))
            ax1.set_xlabel('%s' % (param_label))
            ax1.set_ylabel('Avg Fitness +/- Std Dev', color=color)
            ax1.plot(x, means, linewidth=LINE_WIDTH, markersize=LINE_WIDTH, color=color)
            ax1.fill_between(x, means + stds, means - stds, alpha=0.15, color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.set_ylim([0, max(means+stds) * 1.03])

            ax2 = ax1.twinx()
            color = 'tab:blue';
            ax2.set_ylabel('Count of Function Calls', color=color)
            ax2.plot(x, fn_means, linewidth=LINE_WIDTH, markersize=LINE_WIDTH, color=color)
            ax2.fill_between(x, fn_means + fn_stds, fn_means - fn_stds, alpha=0.15, color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_ylim([0, max(fn_means+fn_stds) * 1.03])

            plt.grid()
            plt.xscale(scale)
            plt.gcf().set_size_inches(FIG_WIDTH, FIG_HEIGHT)
            plt.savefig('tuning_plots/%s_%s_%s.png' % (problem.name, algorithm.name, param_label), bbox_inches='tight')
            plt.close()

        param_ax1.axvline(x=best_value, linestyle='--', linewidth=1, color='black')
        param_ax1.set_ylim(bottom=0.5)
        #param_ax1.tick_params(axis='y')
        param_ax2.set_ylim(bottom=0)
        #param_ax2.tick_params(axis='y')
        #plt.grid()
        param_ax1.legend()
        param_ax1.set_zorder(1)
        param_ax1.set_frame_on(False)
        param_ax2.set_frame_on(True)
        plt.figure(param_fig.number)
        plt.xscale(scale)
        plt.gcf().set_size_inches(FIG_WIDTH, FIG_HEIGHT)
        plt.savefig('tuning_plots/%s_%s.png' % (algorithm.name, param_label), bbox_inches='tight')
        plt.close()

# Based on https://stackoverflow.com/a/50354131
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

# https://matplotlib.org/examples/lines_bars_and_markers/barh_demo.html
def simple_bar_chart(labels, data, err, text_format='%1.2f', **kwargs):
    ax = plt.gca()
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, data, xerr=err)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    label_bars(text_format)

def run_algorithm(problem, algorithm):
    print(problem.name, algorithm.name)
    x_fn, x_iter, iters = run_fitness_by_iteration(problem, DEFAULT_LENGTH, algorithm)
    x, fitnesses, fn_calls, timings = run_fitness_by_length(problem, algorithm)
    return {
            "algorithm": algorithm,
            "x_fn": x_fn,
            "x_iter": x_iter,
            "iters": iters,
            "x": x,
            "fitnesses": fitnesses,
            "fn_calls": fn_calls,
            "timings": timings,
            }


if __name__ == "__main__":
    if not os.path.exists('tuning_plots'):
        os.makedirs('tuning_plots')
    if not os.path.exists('plots'):
        os.makedirs('plots')
    if not os.path.exists('cache'):
        os.makedirs('cache')

    np.random.seed(RANDOM_STATE_OFFSET)

    for problem in FITNESS_FNS:
        fig_num += 1
        fitness_fig = plt.figure(fig_num)
        plt.title('Fitness by length for %s' % (problem.name))

        fig_num += 1
        funcall_fig = plt.figure(fig_num)
        plt.title('Function Calls by length for %s' % (problem.name))

        fig_num += 1
        fn_fig = plt.figure(fig_num)
        plt.title('Fitness by Function Calls for %s (length=%s)' % (problem.name, DEFAULT_LENGTH))

        fig_num += 1
        iter_fig = plt.figure(fig_num)
        plt.title('Fitness by Iterations for %s (length=%s)' % (problem.name, DEFAULT_LENGTH))

        print("Starting %s" % (problem.name))
        # https://medium.com/@mjschillawski/quick-and-easy-parallelization-in-python-32cb9027e490
        results = Parallel()(
                delayed(run_algorithm)(problem, algorithm)
                for algorithm in ALGORITHMS)
        print("Completed %s" % (problem.name))

        #for result in results:
        #    result["x_iter"] *= result["algorithm"].pop_size()

        max_iter_x = max(max(result["x_iter"][:, -1]) for result in results)
        max_fn_x = max(max(result["x_fn"][:, -1]) for result in results)
        bar_labels = []
        bar_timings = []
        bar_stds = []

        for result in results:
            algorithm, x_fn, x_iter, iters, x, fitnesses, fn_calls, timings = result["algorithm"], result["x_fn"], result["x_iter"], result["iters"], result["x"], result["fitnesses"], result["fn_calls"], result["timings"]
            for fig, curr_x, y, y_err, max_x in [(fitness_fig, x, np.mean(fitnesses, axis=1), np.std(fitnesses, axis=1), 0),
                             (funcall_fig, x, np.mean(fn_calls, axis=1), np.std(fn_calls, axis=1), 0),
                             (iter_fig, np.mean(x_iter, axis=0), np.mean(iters, axis=0), np.std(iters, axis=0), max_iter_x),
                             (fn_fig, np.mean(x_fn, axis=0), np.mean(iters, axis=0), np.std(iters, axis=0), max_fn_x)]:
                kwargs = {}
                kwargs["label"] = algorithm.name
                if max_x > 0:
                    kwargs["label"] = "%s (fitness=%d)" % (algorithm.name, y[-1])
                p = fig.gca().plot(curr_x, y, linewidth=LINE_WIDTH, markersize=LINE_WIDTH, **kwargs)
                fig.gca().fill_between(curr_x, y + y_err, y - y_err, alpha=0.15)
                if max_x > 0:
                    curr_max_x = np.mean(curr_x[-1])
                    if curr_max_x < max_x:
                        fig.gca().hlines(y=y[-1], xmin=curr_max_x, xmax=max_x,
                           linestyle=':', linewidths=LINE_WIDTH, colors=p[-1].get_color())

            time_means = np.mean(timings)
            time_stds = np.std(timings)
            bar_labels.append(algorithm.name)
            bar_timings.append(time_means)
            bar_stds.append(time_stds)

        fig_num += 1
        plt.figure(fig_num)
        simple_bar_chart(bar_labels, bar_timings, bar_stds)
        plt.title('Convergence time %s' % (problem.name))
        plt.ylabel('Algorithm')
        plt.xlabel('Time (s)')
        plt.gcf().set_size_inches(FIG_WIDTH, FIG_HEIGHT)
        plt.savefig('plots/%s_%s.png' % (problem.name, 'convergence_times'), bbox_inches='tight')
        plt.close()

        fitness_fig.gca().set_ylabel('Avg Fitness +/- Std Dev')
        fitness_fig.gca().set_xlabel('Problem length')
        funcall_fig.gca().set_ylabel('Count of Function Calls +/- Std Dev')
        funcall_fig.gca().set_xlabel('Problem length')
        fn_fig.gca().set_ylabel('Avg Fitness +/- Std Dev')
        fn_fig.gca().set_xlabel('Count of Function Calls')
        iter_fig.gca().set_ylabel('Avg Fitness +/- Std Dev')
        iter_fig.gca().set_xlabel('Count of Iterations')
        for fig in [fitness_fig, funcall_fig, iter_fig, fn_fig]:
            plt.figure(fig.number)
            plt.grid()
            #plt.ylabel('Avg Score +/- Std Dev')
            #plt.ylim([0, max_fitness * 1.03])
            plt.legend()
            fig.set_size_inches(FIG_WIDTH, FIG_HEIGHT)
            fig.savefig('plots/%s_%s.png' % (problem.name, fig.gca().get_title()), bbox_inches='tight')
            plt.close()

    for algorithm in ALGORITHMS:
        run_fitness_over_param_grid(DEFAULT_LENGTH, algorithm)

