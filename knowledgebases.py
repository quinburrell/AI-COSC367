import re
import term1labs

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