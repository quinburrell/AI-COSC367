"""
Assignment 2 COSC 367
This program takes a sequence of numbers that represent a simle mathematical 
pattern, then attempts to predict the next 5 numbers in the sequence.
"""
import random

def is_valid_expression(object, function_symbols, leaf_symbols):
    """Outputs a boolean indicating the validity of the input object"""
    if type(object) == list:
        if len(object) != 3 or object[0] not in function_symbols \
                or not \
                is_valid_expression(object[1], function_symbols, leaf_symbols) \
                or not \
                is_valid_expression(object[2], function_symbols, leaf_symbols):
            return False
    elif type(object) != int and object not in leaf_symbols:
        return False
    return True


def depth(expression, count = 0):
    if type(expression) == list:
        count = max(depth(obj) for obj in expression) + 1
    return count


def evaluate(expression, bindings):
    if type(expression) == str:
        return bindings[expression]
    elif type(expression) == int:
        return expression
    elif type(expression) == list:
        return bindings[expression[0]](evaluate(expression[1], bindings), evaluate(expression[2], bindings))


def random_expression(function_symbols, leaves, max_depth):
    if random.getrandbits(1) and max_depth > 0:
        expression = list(function_symbols[random.randint(0, len(function_symbols)-1)])
        expression.append(random_expression(function_symbols, leaves, max_depth - 1))
        while len(expression) < 3:
            expression.append(random_expression(function_symbols, leaves, max_depth-1))
    else:
        expression = leaves[random.randint(0, len(leaves) - 1)]
    return expression


def generate_rest(initial_sequence, expression, length):
    if length == 0:
        return []
    else:
        bindings = {'i': len(initial_sequence),
                    '+': lambda x, y: x + y, '*': lambda x, y: x * y, '-': lambda x, y: x - y}
        if bindings['i'] > 1:
            bindings['x'] = initial_sequence[-2]
            bindings['y'] = initial_sequence[-1]
        next = [evaluate(expression, bindings)]
        return next + generate_rest(initial_sequence + next, expression, length - 1)


def predict_rest(sequence):
    function_symbols = ['*', '+', '-']
    leaves = ['i', 'x', 'y'] + list(range(-2, 2))
    expressions = [random_expression(function_symbols, leaves, 3) for _ in range(10000)]
    bindings = {'+': lambda x, y: x + y, '*': lambda x, y: x * y, '-': lambda x, y: x - y}
    for expression in expressions:
        i = 2
        while i < len(sequence):
            bindings['i'] = i
            bindings['x'] = sequence[i-2]
            bindings['y'] = sequence[i-1]
            if sequence[i] != evaluate(expression, bindings):
                break
            else:
                i += 1
        if i == len(sequence):
            break
    print(expression)
    return generate_rest(sequence, expression, 5)

random.seed(79)
sequence = [0, 1, 2, 3, 4, 5, 6, 7]
the_rest = predict_rest(sequence)
print('seq:', sequence)
print(the_rest)


sequence = [0, 2, 4, 6, 8, 10, 12, 14]
print('seq:', sequence)
print(predict_rest(sequence))


sequence = [31, 29, 27, 25, 23, 21]
print('seq:', sequence)
print(predict_rest(sequence))


sequence = [0, 1, 4, 9, 16, 25, 36, 49]
print('seq:', sequence)
print(predict_rest(sequence))


sequence = [3, 2, 3, 6, 11, 18, 27, 38]
print('seq:', sequence)
print(predict_rest(sequence))


sequence = [0, 1, 1, 2, 3, 5, 8, 13]
print('seq:', sequence)
print(predict_rest(sequence))


sequence = [0, -1, 1, 0, 1, -1, 2, -1]
print('seq:', sequence)
print(predict_rest(sequence))


sequence = [1, 3, -5, 13, -31, 75, -181, 437]
print('seq:', sequence)
print(predict_rest(sequence))
