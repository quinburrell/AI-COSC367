import random
import itertools
import math
import csv


def n_queens_neighbours(state):
    res = []
    for q in range(len(state)):
        for p in range(q, len(state)):
            if p != q:
                statelist = list(state)
                statelist[p], statelist[q] = statelist[q], statelist[p]
                res.append(tuple(statelist))
    return sorted(res)


def n_queens_cost(state):
    cost = 0
    for q in range(len(state)):
        for p in range(q, len(state)):
            if q != p:
                dx = q - p
                dy = state[q] - state[p]
                if abs(dx) == abs(dy):
                    cost += 1
    return cost


def greedy_descent(initial_state, neighbours, cost):
    res = [initial_state]
    nbs = {}
    for neighbour in neighbours(initial_state):
        if cost(neighbour) not in nbs:
            nbs[cost(neighbour)] = neighbour
    new = min(nbs.keys()) if nbs != {} else None
    if new is not None and new < cost(initial_state):
        res += greedy_descent(nbs[new], neighbours, cost)
    return res


def greedy_descent_with_random_restart(random_state, neighbours, cost):
    state = random_state()
    for path in greedy_descent(state, neighbours, cost):
        print(path)
    if cost(path) != 0:
        print('RESTART')
        greedy_descent_with_random_restart(random_state, neighbours, cost)


def roulette_wheel_select(population, fitness, r):
    fitsum = sum(fitness(pop) for pop in population)
    for pop in population:
        r -= fitness(pop)/fitsum
        if r <= 0:
            return pop


def joint_prob(network, assignment):
    p = 1
    for var in network:
        boochain = []
        for parent in network[var]['Parents']:
            boochain += [assignment[parent]]
        probvar = network[var]['CPT'][tuple(boochain)]
        if not assignment[var]:
            probvar = 1 - probvar
        p = p * probvar
    return p


def query(network, query_var, evidence):
    res = {}
    assignment = dict(evidence)
    hidden_vars = network.keys() - evidence.keys() - {query_var}
    for query_value in {True, False}:
        rawdist = []
        assignment[query_var] = query_value
        for values in itertools.product((True, False), repeat=len(hidden_vars)):
            assignment.update({var: val for var, val in zip(hidden_vars, values)})
            rawdist += [joint_prob(network, assignment)]
        res[query_value] = sum(rawdist)
    res[True], res[False] = res[True] / (res[True] + res[False]), res[False] / (res[True] + res[False])
    return res


def posterior(prior, likelihood, observation):
    p_true = prior
    p_false = 1 - prior
    for i in range(len(observation)):
        if observation[i]:
            p_true *= likelihood[i][True]
            p_false *= likelihood[i][False]
        else:
            p_true *= 1 - likelihood[i][True]
            p_false *= 1 - likelihood[i][False]
    return p_true/(p_true+p_false)


def learn_prior(file_name, pseudo_count=0):
    with open(file_name) as in_file:
        training_examples = [tuple(row) for row in csv.reader(in_file)]
    return (sum(int(training_examples[i][-1]) for i in range(1, len(training_examples))) + pseudo_count)\
        / ((len(training_examples) - 1) + pseudo_count*2)


def learn_likelihood(file_name, pseudo_count=0, true_count=0):
    with open(file_name) as in_file:
        training_examples = [tuple(row) for row in csv.reader(in_file)]
    result = []
    for i in range(len(training_examples[0]) - 1):
        result += [[0, 0]]
    for ass in training_examples[1:]:
        true_count += int(ass[-1])
        for i in range(len(ass) - 1):
            result[i][int(ass[-1])] = result[i][int(ass[-1])] + int(ass[i])
    for i in range(len(result)):
        try:
            p_false = (result[i][0] + pseudo_count)/((len(training_examples[1:]) - true_count) + pseudo_count*2)
        except ZeroDivisionError:
            p_false = 0
        try:
            p_true = (result[i][1] + pseudo_count)/(true_count + pseudo_count*2)
        except ZeroDivisionError:
            p_true = 0
        result[i] = p_false, p_true
    return result


def nb_classify(prior, likelihood, input_vector):
    p_spam = posterior(prior, likelihood, input_vector)
    if p_spam >= 0.5:
        return "Spam", p_spam
    else:
        return "Not Spam", (1 - p_spam)


def euclidean_distance(v1, v2):
    if len(v1) == len(v2):
        dist = sum((v1[i] - v2[i])**2 for i in range(len(v1)))
        return math.sqrt(abs(dist))


def majority_element(labels):
    res_dict = {}
    for label in labels:
        if label in res_dict:
            res_dict[label] += 1
        else:
            res_dict[label] = 1
    return sorted(res_dict, key=res_dict.get)[-1]


def construct_perceptron(weights, bias):
    """Returns a perceptron function using the given paramers."""
    def perceptron(input):
        if bias + sum(input[i] * weights[i] for i in range(len(input))) < 0:
            return 0
        return 1
    return perceptron


def accuracy(classifier, inputs, expected_outputs):
    return sum(classifier(inputs[i]) == expected_outputs[i] for i in range(len(inputs))) / len(inputs)


def learn_perceptron_parameters(weights, bias, training_examples, learning_rate, max_epochs):
    epochs = 0
    perceptron = construct_perceptron(weights, bias)
    while epochs < max_epochs:
        epochs += 1
        no_changes = 1
        for example in training_examples:
            inputs, expected = example
            inaccuracy = expected - perceptron(inputs)
            if inaccuracy != 0:
                no_changes = 0
                for i in range(len(weights)):
                    weights[i] = weights[i] + (learning_rate * inputs[i] * inaccuracy)
                bias = bias + (learning_rate * inaccuracy)
                perceptron = construct_perceptron(weights, bias)
        if no_changes:
            return weights, bias
    return weights, bias


def max_value(tree):
    if type(tree) is list:
        temp = list()
        for i in range(len(tree)):
            if type(tree[i]) is list:
                temp.append(min_value(tree[i]))
            else:
                temp.append(tree[i])
        return max(temp)
    else:
        return tree


def min_value(tree):
    if type(tree) is list:
        temp = list()
        for i in range(len(tree)):
            if type(tree[i]) is list:
                temp.append(max_value(tree[i]))
            else:
                temp.append(tree[i])
        return min(temp)
    else:
        return tree


def alpha_beta_prune(alpha, beta, max_min):
    if type(beta) is int:
        return alpha, beta
    for beta_entry in beta:
        if beta_entry > alpha:
            alpha = beta_entry
            if max_min:
                break


def max_action_value(game_tree):
    if type(game_tree) is int:
        return None, game_tree
    temp = list()
    for i in range(len(game_tree)):
        if type(game_tree[i]) is list:
            temp.append(min_value(game_tree[i]))
        else:
            temp.append(game_tree[i])
    return temp.index(max(temp)), max(temp)


def min_action_value(game_tree):
    if type(game_tree) is int:
        return None, game_tree
    temp = list()
    for i in range(len(game_tree)):
        if type(game_tree[i]) is list:
            temp.append(max_value(game_tree[i]))
        else:
            temp.append(game_tree[i])
    return temp.index(min(temp)), min(temp)


game_tree = [3, [[2, 1], [4, [7, -2]]], 0]

max_action_value(game_tree)


