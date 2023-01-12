import cvxpy as cp
from collections import defaultdict
import numpy as np

def create_basic_mip(scores, final_num_topics):
    
    """Abstraction of the base MIP model

    Parameters
    ----------
    scores : list of float
        Fitness score to be used for decisions
    final_topic_num : int
        (K) Number of chosen topics

    Returns
    -------
    cvpxy objective
    cvpxy constraints
    cvpxy variables
    
    """
    
    x_vars = cp.Variable(len(scores), boolean=True)
    constraints = [cp.sum(x_vars) == final_num_topics]
    objective = cp.Maximize(cp.sum(scores @ x_vars))
    
    return objective, constraints, x_vars

def mdkp(topics, scores, final_num_topics, epsilon,
         solver=None, solver_options=None, **kwargs):
    
    """Multi-Dimensional Knapsack Problem (MDKP) 
    formulation for composite topic selection

    Parameters
    ----------
    topics : list of list of identifier tokens
        Tokens can represent word or vocab identifiers
    scores : list of float
        Fitness score to be used for decisions
    final_topic_num : int
        (K) Number of chosen topics
    epsilon : int
        Hyper-parameter controlling uniqueness of topics. 
        Set 0 for unique topics.
        Setting higher values might decrease uniqueness.
    solver : supported solver in cvxpy, optional
        defaults to cp.GLPK_MI
    solver_options : dictionary to be passed into solver
        defaults to default values of cvxpy:
        https://www.cvxpy.org/tutorial/advanced/index.html#setting-solver-options

    Returns
    -------
    1-d numpy boolean array
        indicating which topics are chosen
    """
    if not solver:
        solver = cp.GLPK_MI
    if not solver_options:
        solver_options = {}
        
    topics = [set(t) for t in topics]

    words = defaultdict(set)
    for i in range(len(topics)):
        for j in range(i+1, len(topics)):
            for word in topics[i].intersection(topics[j]):
                words[word].add(i)
                words[word].add(j)

    objective, constraints, x_vars = create_basic_mip(scores, final_num_topics)
    
    w_vars = cp.Variable(len(words), boolean=True)
    constraints += [cp.sum(w_vars) >= epsilon]
    
    for i, w in enumerate(words):
        constraints += [w_vars[i] <= cp.sum([x_vars[j] for j in words[w]])]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=True, **solver_options)
    
    return np.array(x_vars.value, dtype=bool)

def mwbis(topics, scores, final_num_topics, epsilon,
          solver=None, solver_options=None, **kwargs):
    
    """Maximum-Weight Budget Independent Set (MWBIS) 
    formulation for composite topic selection

    Parameters
    ----------
    topics : list of list of identifier token
        Tokens can represent word or vocab identifiers
    scores : list of float
        Fitness score to be used for decisions
    final_topic_num : int
        (K) Number of chosen topics
    epsilon : int
        Hyper-parameter controlling uniqueness of topics. 
        Set 0 for unique topics.
        Setting higher values might decrease uniqueness.
    solver : supported solver in cvxpy, optional
        defaults to cp.GLPK_MI
    solver_options : dictionary to be passed into solver
        defaults to default values of cvxpy:
        https://www.cvxpy.org/tutorial/advanced/index.html#setting-solver-options

    Returns
    -------
    1-d numpy boolean array
        indicating which topics are chosen
    """
    if not solver:
        solver = cp.GLPK_MI
    if not solver_options:
        solver_options = {}
        
    topics = [set(t) for t in topics]

    words = defaultdict(set)
    edges = []
    for i in range(len(topics)):
        for j in range(i+1, len(topics)):
            num_duplicates = len(topics[i]) + len(topics[j])  \
            - len(topics[i].union(topics[j]))
            
            if num_duplicates > epsilon:
                edges.append((i,j))
            for word in topics[i].intersection(topics[j]):
                words[word].add(i)
                words[word].add(j)
    
    objective, constraints, x_vars = create_basic_mip(scores, final_num_topics)
    
    for w in words:
        constraints += [cp.sum([x_vars[i] for i in words[w]]) <= epsilon + 1]
    for i,j in edges:
        constraints += [x_vars[i] + x_vars[j] <= 1]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=True, **solver_options)
    
    return np.array(x_vars.value, dtype=bool)
