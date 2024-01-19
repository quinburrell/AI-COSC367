from search import *
from collections import deque
import itertools
import copy
import math
import re
import statistics


class DFSFrontier():
    """Implements a frontier container appropriate for depth-first
    search."""

    def __init__(self):
        """The constructor takes no argument. It initialises the
        container to an empty stack."""
        self.container = []

    def add(self, path):
        self.container.append(path)

    def __iter__(self):
        """The object returns itself because it is implementing a __next__
        method and does not need any additional state for iteration."""
        return self

    def __next__(self):
        if len(self.container) > 0:
            return self.container.pop()
        else:
            raise StopIteration  # empty


class BFSFrontier():
    """Implements a frontier container appropriate for breadth-first
    search."""

    def __init__(self):
        """The constructor takes no argument. It initialises the
        container to an empty stack."""
        self.container = deque([])

    def add(self, path):
        self.container.append(path)

    def __iter__(self):
        """The object returns itself because it is implementing a __next__
        method and does not need any additional state for iteration."""
        return self

    def __next__(self):
        if len(self.container) > 0:
            return self.container.popleft()
        else:
            raise StopIteration  # empty


class FunkyNumericGraph():
    """A graph where nodes are numbers. A number n leads to n-1 and
    n+2. Nodes that are divisible by 10 are goal nodes."""

    def __init__(self, starting_number):
        self.starting_number = starting_number

    def outgoing_arcs(self, tail_node):
        """Takes a node (which is an integer in this problem) and returns
        outgoing arcs (always two arcs in this problem)"""
        return [Arc(tail_node, tail_node - 1, action="1down", cost=1),
                Arc(tail_node, tail_node + 2, action="2up", cost=1)]

    def starting_nodes(self):
        """Returns a sequence (list) of starting nodes. In this problem
        the seqence always has one element."""
        return [self.starting_number]

    def is_goal(self, node):
        """Determine whether a given node (integer) is a goal."""
        return node % 10 == 0


class SlidingPuzzleGraph():
    """Objects of this type represent (n squared minus one)-puzzles.
    """

    def __init__(self, starting_state):
        self.starting_state = starting_state

    def outgoing_arcs(self, state):
        """Given a puzzle state (node) returns a list of arcs. Each arc
        represents a possible action (move) and the resulting state."""

        n = len(state)  # the size of the puzzle
        i, j = next((i, j) for i in range(n) for j in range(n)
                    if state[i][j] == BLANK)  # find the blank tile
        arcs = []
        if i > 0:
            action = "Move {} down".format(state[i - 1][j])  # or blank goes up
            new_state = copy.deepcopy(state)
            new_state[i][j], new_state[i - 1][j] = new_state[i - 1][j], BLANK
            arcs.append(Arc(state, new_state, action, 1))
        if i < n - 1:
            action = "Move {} up".format(state[i + 1][j])  # or blank goes down
            new_state = copy.deepcopy(state)
            new_state[i][j], new_state[i + 1][j] = new_state[i + 1][j], BLANK
            arcs.append(Arc(state, new_state, action, 1))
        if j > 0:
            action = "Move {} right".format(state[i][j - 1])  # or blank goes left
            new_state = copy.deepcopy(state)
            new_state[i][j], new_state[i][j - 1] = new_state[i][j - 1], BLANK
            arcs.append(Arc(state, new_state, action, 1))
        if j < n - 1:
            action = "Move {} left".format(state[i][j + 1])  # or blank goes right
            new_state = copy.deepcopy(state)
            new_state[i][j], new_state[i][j + 1] = new_state[i][j + 1], BLANK
            arcs.append(Arc(state, new_state, action, 1))
        return arcs

    def starting_nodes(self):
        return [self.starting_state]

    def is_goal(self, state):
        """Returns true if the given state is the goal state, False
        otherwise. There is only one goal state in this problem."""

        n = len(state)
        check = 1
        i, j = 0, 1
        while i < n:
            while j < n:
                if state[i][j] != check:
                    return False
                else:
                    check += 1
                    j += 1
            i, j = i + 1, 0
        return True


class LocationGraph():
    """A graph class of locations on a grid
    """
    def __init__(self, nodes, locations, edges, starting_nodes, goal_nodes):
        self.nodes = nodes
        self.locations = locations
        self.edge_list = edges
        self._starting_nodes = starting_nodes
        self.goal_nodes = goal_nodes

    def starting_nodes(self):
        """Returns a sequence of starting nodes."""
        return self._starting_nodes

    def is_goal(self, node):
        """Returns true if the given node is a goal node."""
        return node in self.goal_nodes

    def outgoing_arcs(self, node):
        """Returns a sequence of Arc objects that go out from the given
        node. The action string is automatically generated.

        """
        arcs = []
        for edge in self.edge_list:
            tail, head = edge
            dx, dy = self.locations[head][0] - self.locations[tail][0], self.locations[head][1] - self.locations[tail][1]
            cost = math.sqrt(dx**2 + dy**2)

            if tail == node:
                arcs.append(Arc(tail, head, str(tail) + '->' + str(head), cost))
            elif head == node:
                arcs.append(Arc(head, tail, str(head) + '->' + str(tail), cost))

        def sort(arcs):
            temp = []
            for arc in arcs:
                if temp == []:
                    temp.append(arc)
                else:
                    i = 0
                    while i < len(temp) and arc.tail + arc.head > temp[i].tail + temp[i].head:
                        i += 1
                    if i == len(temp) or arc != temp[i]:
                        temp = temp[:i] + [arc] + temp[i:]

            return temp

        return sort(arcs)


class LCFSFrontier(Frontier):
    """A frontier container for Lowest Cost First Searched"""
    def __init__(self):
        self.container = []

    def add(self, path):
        pathcost = sum(arc.cost for arc in path)
        i = 0
        while i < len(self.container) and pathcost < sum(arc.cost for arc in self.container[i]):
            i += 1
        self.container = self.container[:i] + [path] + self.container[i:]

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.container) > 0:
            return self.container.pop()
        else:
            raise StopIteration  # empty


def clauses(knowledge_base):
    """Takes the string of a knowledge base; returns an iterator for pairs
    of (head, body) for propositional definite clauses in the
    knowledge base. Atoms are returned as strings. The head is an atom
    and the body is a (possibly empty) list of atoms.

    -- Kourosh Neshatian - 2 Aug 2021

    """
    ATOM   = r"[a-z][a-zA-Z\d_]*"
    HEAD   = rf"\s*(?P<HEAD>{ATOM})\s*"
    BODY   = rf"\s*(?P<BODY>{ATOM}\s*(,\s*{ATOM}\s*)*)\s*"
    CLAUSE = rf"{HEAD}(:-{BODY})?\."
    KB     = rf"^({CLAUSE})*\s*$"

    assert re.match(KB, knowledge_base)

    for mo in re.finditer(CLAUSE, knowledge_base):
        yield mo.group('HEAD'), re.findall(ATOM, mo.group('BODY') or "")


def forward_deduce(kb, change=1):
    """takes a knowledge base string and derives a set of true atoms from it"""
    result = []
    while change == 1:
        change = 0
        for clause in clauses(kb):
            if clause[1] is [] or all(atom in result for atom in clause[1]):
                if clause[0] not in result:
                    change = 1
                    result.append(clause[0])
    return result


class KBGraph():
    def __init__(self, kb, query):
        self.clauses = list(clauses(kb))
        self.query = query

    def starting_nodes(self):
        return [list(self.query)]

    def is_goal(self, node):
        return node == []

    def outgoing_arcs(self, tail_node):
        arcs = []
        for clause in self.clauses:
            for i in range(len(tail_node)):
                if clause[0] == tail_node[i]:
                    arcs.append(Arc(tail_node, clause[1] + tail_node[:i] + tail_node[i+1:], 'nada', 1))
        return arcs


class PriorityFrontier(Frontier):

    def __init__(self):
        # The constructor does not take any arguments.
        self.container = []
        # Complete the rest

    def add(self, path):
        if len(path) < 20:
            pathcost = sum(arc.cost for arc in path)
            pvar = statistics.pvariance([arc.cost for arc in path])
            i = 0
            while i < len(self.container) and pathcost <= sum(arc.cost for arc in self.container[i]) and pvar < statistics.pvariance([arc.cost for arc in self.container[i]]):
                i += 1
            self.container = self.container[:i] + [path] + self.container[i:]

    def __iter__(self):
        """The object returns itself because it is implementing a __next__
        method and does not need any additional information for iteration."""
        return self

    def __next__(self):
        if len(self.container) > 0:
            return self.container.pop()
        else:
            raise StopIteration

graph = ExplicitGraph(
    nodes = {'S', 'A', 'B', 'G'},
    edge_list=[('S','A', 1), ('A', 'B', 1), ('B','S',1)],
    starting_nodes = ['S'],
    goal_nodes = {'G'})

solution = next(generic_search(graph, PriorityFrontier()), None)
print_actions(solution)
