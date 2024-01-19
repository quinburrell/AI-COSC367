import random
import itertools
import math
import csv

def roulette_wheel_select(population, fitness, r):
    fitsum = sum(fitness(pop) for pop in population)
    for pop in population:
        r -= fitness(pop)/fitsum
        if r <= 0:
            return pop


def select(population, error, max_error, r):
    fitsum = sum((max_error - error(pop)) for pop in population)
    for pop in population:
        r -= (max_error - error(pop))/fitsum
        if r < 0:
            return pop


def estimate(time, observations, k):
    keep = dict()
    for i in range(len(observations)):
        keep[i] = time - observations[i][0]
    av = 0
    for x in range(k):
        av += observations[sorted(keep.values())[x]][1]
    return av/k


def num_parameters(unit_counts):
    return sum(unit_counts[1:]) + sum(unit_counts[i] * unit_counts[i-1] for i in range(1, len(unit_counts)))

def num_crossovers(expression1, expression2):
    return len(expression1) ** 2

expression1 = ['+', 12, 'x']
expression2 = ['-', 3, 6]
print(num_crossovers(expression1, expression2))