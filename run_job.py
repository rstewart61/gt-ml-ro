#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 14:56:00 2020

@author: brandon
"""

import numpy as np
import traceback

import sys

import time

import mlrose_hiive as mlrose # doesn't have fast_mimic

# TODO: Reenable grid param plots
# TODO: Compare wall times.
# TODO: Improve fn_call counts

"""
Genetic: *Four Peaks, Traveling Sales
Simulated Annealing: *Continuous Peaks
MIMIC: *Knapsack, Max K-Color
"""

MAX_ITERS = 40000
RANDOM_STATE_OFFSET = 3000


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

    def instance(self, problem, **kwargs):
        args = {**self.default_args, **kwargs}
        schedule_args = {}
        for k in ['init_temp', 'decay', 'min_temp']:
            if k in args.keys():
                schedule_args[k] = args[k]
                del args[k]
        if schedule_args:
            args['schedule'] = NamedGeomDecay(**schedule_args)
        return self.algorithm(problem, **args)

    def pop_size(self):
        if 'pop_size' in self.default_args:
            return self.default_args['pop_size']
        return 1

    def __repr__(self):
        return "%s" % (self.name)

    def __getstate__(self):
        return (self.name, self.algorithm.__name__, self.param_grid, self.default_args)

    def __setstate__(self, state):
        algorithm = getattr(mlrose, state[1])
        Algorithm.__init__(self, state[0], algorithm, state[2], **state[3])

simulated_annealing_grid = {
        #'schedule': [mlrose.GeomDecay, mlrose.ArithDecay, mlrose.ExpDecay]
        'max_attempts': (range(1, 51, 2), 'linear'),
        'decay': (1-np.logspace(-0.1, -9, 30), 'log'),
        'init_temp': (np.logspace(-3, 4, 20), 'log'),
        'min_temp': (np.logspace(-1, -7, 20), 'log'),
        }

rhc_grid = {
        'max_attempts': (range(1, 51, 2), 'linear'),
        'restarts': (range(1, 51, 2), 'linear'),
        }

genetic_grid = {
        'pop_size': (range(50, 701, 50), 'linear'),
        #'mutation_prob': (np.logspace(-2, -0.01, 20), 'log'),
        'mutation_prob': (np.linspace(0.05, 0.95, 19), 'linear'),
        #'mutation_prob': (np.linspace(0.1, 0.9, 20), 'linear'),
        'max_attempts': (range(1, 51, 2), 'linear'),
        }

mimic_grid = {
        #'keep_pct': (np.logspace(-2, -0.01, 20), 'log'),
        'pop_size': (range(50, 751, 50), 'linear'),
        #'keep_pct': (np.linspace(0.1, 0.9, 20), 'linear'),
        'keep_pct': (np.linspace(0.05, 0.95, 19), 'linear'),
        'max_attempts': (range(1, 51, 2), 'linear'),
        }

ALGORITHMS = [
    Algorithm('RHC', mlrose.random_hill_climb, rhc_grid,
              max_iters=MAX_ITERS, max_attempts=40, restarts=20),
    Algorithm('SA', mlrose.simulated_annealing, simulated_annealing_grid,
                                     init_temp=1.0, decay=1e-4, min_temp=0.001, max_attempts=40,
                                     max_iters=MAX_ITERS),
    Algorithm('GA', mlrose.genetic_alg, genetic_grid, max_iters=MAX_ITERS/250,
              max_attempts=40, pop_size=200, mutation_prob=0.1),
    Algorithm('MIMIC', mlrose.mimic, mimic_grid, max_iters=MAX_ITERS/350, keep_pct=0.20,
              max_attempts=40, pop_size=700),
]

def four_peaks(length=50):
    return mlrose.FourPeaks(), length*2

def six_peaks(length=50):
    return mlrose.SixPeaks(), length*2

def continuous_peaks(length=50):
    return mlrose.ContinuousPeaks(), length*2

def max_k_color(length=50):
    v1 = np.random.randint(length, size=length)
    v2 = np.random.randint(length, size=length)
    for i in range(len(v1)):
        while v1[i] == v2[i]:
            v1[i] = np.random.randint(length)
    edges = list(zip(v1, v2))
    return mlrose.MaxKColor(edges), length

def travelling_sales(length=50):
    dists = []
    n = length
    for i in range(n):
        for j in range(i+1, n):
            if np.random.random_sample() < 1.0:
                l = np.random.randint(1, 100)
                dists.append((i, j, l))
    return mlrose.TravellingSales(distances=dists), length*100

def knapsack(length=100, pct=0.5):
    weights = np.random.randint(1, length, size=length)
    values = np.random.randint(1, length, size=length)
    return mlrose.Knapsack(weights, values, pct), sum(values)

def queens(length=10):
    return mlrose.Queens(), length*(length-1)/2

class FitnessCounter:
    def __init__(self, fitness_fn):
        self.fitness_fn = fitness_fn
        self.n_evaluations = 0

    def evaluate(self, state):
        self.n_evaluations += 1
        return self.fitness_fn.evaluate(state)

    def get_prob_type(self):
        return self.fitness_fn.get_prob_type()

class Fitness:
    def __init__(self, name, fn, opt):
        self.name = name
        self.fn = fn
        self.opt = opt

    def get_count(self):
        return self.fitness.n_evaluations

    """
    def instance(self, length):
        m = {'max_k_color': max_k_color,
             'travelling_sales': travelling_sales,
             'continuous_peaks': continuous_peaks
                }
        fn = m[self.fn]
        fitness, normalization = fn(length)
        self.fitness = FitnessCounter(fitness)
        return self.opt(length=length, fitness_fn=self.fitness, maximize=True), normalization
    """

    def __repr__(self):
        return "%s" % (self.name)

def instance(problem, length):
    m = {'max_k_color': max_k_color,
         'travelling_sales': travelling_sales,
         'continuous_peaks': continuous_peaks,
         'four_peaks': four_peaks,
         'knapsack': knapsack,
            }
    fn = m[problem.fn]
    fitness, normalization = fn(length)
    problem.fitness = FitnessCounter(fitness)
    return problem.opt(length=length, fitness_fn=problem.fitness, maximize=True), normalization


FITNESS_FNS = [
    Fitness('Traveling Sales', 'travelling_sales', mlrose.TSPOpt),
    Fitness('Max K-Color', 'max_k_color', mlrose.DiscreteOpt),
    Fitness('Continuous Peaks', 'continuous_peaks', mlrose.DiscreteOpt),
    #Fitness('Four Peaks', 'four_peaks', mlrose.DiscreteOpt),
    #Fitness('Knapsack', 'knapsack', mlrose.DiscreteOpt),
    #Fitness('Six Peaks', six_peaks, mlrose.DiscreteOpt),
    #Fitness('Queens', queens, mlrose.DiscreteOpt),
]

def run_iteration(i, problem_name, length, algorithm_name, **kwargs):
    try:
        for j in range(len(FITNESS_FNS)):
            if FITNESS_FNS[j].name == problem_name:
                problem = FITNESS_FNS[j]
        for j in range(len(ALGORITHMS)):
            if ALGORITHMS[j].name == algorithm_name:
                algorithm = ALGORITHMS[j]

        opt, normalization = instance(problem, length)
        if algorithm_name == 'MIMIC':
            #kwargs['fast_mimic'] = True
            #kwargs['fast_mimic'] = (problem.name != 'Traveling Sales')
            #opt.set_mimic_fast_mode(problem.name != 'Traveling Sales')
            opt.set_mimic_fast_mode(True)
            pass
        seed = RANDOM_STATE_OFFSET + i * length
        start = time.time()
        state, fitness, curve = algorithm.instance(opt, max_iters=MAX_ITERS,
                                                   random_state=seed, curve=True,
                                                   **kwargs)
        time_taken = time.time() - start
        curr_iters = len(curve)
        results = np.zeros((MAX_ITERS))
        results[:curr_iters] = curve
        for j in range(curr_iters, MAX_ITERS):
            results[j] = fitness
        return fitness, curr_iters, results, problem.get_count(), time_taken, normalization
    except:
        raise Exception(str(kwargs) + "".join(traceback.format_exception(*sys.exc_info())))