from search import *
import math
import heapq
import itertools
counter = itertools.count()

class RoutingGraph(Graph):
    """graph representation of routes around a map"""
    def __init__(self, map_str):
        map_str = map_str.split('\n')
        for i, row in enumerate(map_str):
            map_str[i] = row.strip()
        goals = []
        agents = []
        for x, row in enumerate(map_str):
            row = row.strip()
            for y, c in enumerate(row):
                if c == 'G':
                    goals.append((x, y))
                elif c == 'S':
                    agents.append((x, y, math.inf))
                elif c.isdigit():
                    agents.append((x, y, int(c)))
        self.agents = agents
        self.goals = goals
        self.map = map_str

    def starting_nodes(self):
        return self.agents

    def outgoing_arcs(self, tail_node):
        arcs = []
        row, col, fuel = tail_node
        dirs = [('N', -1, 0), ('E', 0, 1), ('S', 1, 0), ('W', 0, -1)]
        for dir in dirs:
            new_row, new_col = row + dir[1], col + dir[2]
            new_loc = self.map[new_row][new_col]
            if new_loc != 'X' and new_loc != '-' and new_loc != '|'and fuel > 0:
                arcs.append(Arc(tail_node, (new_row, new_col, fuel-1), dir[0], 5))
        if self.map[row][col] == 'F' and fuel < 9:
            arcs.append(Arc(tail_node, (row, col, 9), 'Fuel up', 15))
        return arcs

    def is_goal(self, node):
        return (node[0], node[1]) in self.goals

    def estimated_cost_to_goal(self, path):
        node = path[-1].head
        cost = min(abs(node[0] - goal[0]) + abs(node[1] - goal[1]) for goal in self.goals) * 5
        if cost/5 > node[2]:
            cost += 15
        return cost + sum(arc.cost for arc in path)


class AStarFrontier(Frontier):
    """A frontier class for A star routing in a grid world"""
    def __init__(self, map):
        self.container = []
        self.map = map
        self.explored = set()

    def add(self, path):
        if path[-1].head not in self.explored:
            heapq.heappush(self.container, (self.map.estimated_cost_to_goal(path), next(counter), path))

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.container) > 0:
            path = heapq.heappop(self.container)[2]
            self.explored.add(path[-1].head)
            return path
        else:
            raise StopIteration


def print_map(graph, frontier, solution):
    """prints a map displaying the solution to a gridworld routing problem"""
    map_list = []
    for row in graph.map:
        map_list += [list(row)]
    if solution is not None:
        for arc in solution:
            row, col, fuel = arc.head
            if map_list[row][col] == ' ':
                map_list[row][col] = '*'
    for tile in frontier.explored:
        row, col, fuel = tile
        if map_list[row][col] == ' ':
            map_list[row][col] = '.'
    for i, row in enumerate(map_list):
        map_list[i] = ''.join(row)
    print('\n'.join(map_list))

