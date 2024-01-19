from csp import *
import itertools, copy


def generate_and_test(csp):
    names, domains = zip(*csp.var_domains.items())
    for values in itertools.product(*domains):
        assignment = dict(zip(names, values))
        if all(satisfies(assignment, constraint) for constraint in csp.constraints):
            yield assignment


def arc_consistent(csp):
    csp = copy.deepcopy(csp)
    to_do = {(x, c) for c in csp.constraints for x in csp.var_domains.keys()}
    while to_do:
        x, c = to_do.pop()
        ys = scope(c) - {x}
        new_domain = set()
        for xval in csp.var_domains[x]:
            assignment = {x: xval}
            for yvals in itertools.product(*[csp.var_domains[y] for y in ys]):
                assignment.update({y: yval for y, yval in zip(ys, yvals)})
                if satisfies(assignment, c):
                    new_domain.add(xval)
                    break
        if csp.var_domains[x] != new_domain:
            for cprime in set(csp.constraints) - {c}:
                if x in scope(cprime):
                   for z in scope(cprime):
                       if x != z:
                           to_do.add((z, cprime))
            csp.var_domains[x] = new_domain
    return csp


csp = CSP(
   var_domains = {var:{-1,0,1} for var in 'abcd'},
   constraints = {
      lambda a, b: a == abs(b),
      lambda c, d: c > d,
      lambda a, b, c: a * b > c + 1
      }
   )


relations = [
    Relation(
        header=['a', 'b'],
        tuples={(0, 0),
                (1, 1),
                (1, -1)}
    ),
    Relation(
        header=['a', 'b', 'c'],
        tuples={(-1, -1, -1),
                (1, 1, -1)}
    ),
    Relation(
        header=['c', 'd'],
        tuples={(0, -1),
                (1, -1),
                (1, 0)}
    )
]

relations_after_elimination = [
    Relation(
        header=['a', 'b'],
        tuples={(1, 1)}
    ),
    Relation(
        header=['a', 'b', 'c'],
        tuples={(1, 1, -1)}
    )
]

print(len(relations))
print(all(type(r) is Relation for r in relations))