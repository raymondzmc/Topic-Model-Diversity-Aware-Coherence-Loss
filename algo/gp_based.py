import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
import numpy as np

def create_basic_mip(scores, final_num_topics, warm_start_indices, 
                     MIP_gap, time_limit):
    
    """Abstraction of the base MIP model

    Parameters
    ----------
    scores : list of float
        Fitness score to be used for decisions
    final_topic_num : int
        (K) Number of chosen topics
    warm_start_indices : list of int
        indices for starting solution
    MIP_gap : float
        Termination criteria for solution search
        Set 0 to find optimal solution (might be slower)
    time_limit : int
        Time (seconds) before stopping search for solution 

    Returns
    -------
    Gurobipy MIP model
    list of model binary variables denoting topic choice
    
    """
    m = gp.Model("mip")
    x_vars = [m.addVar(vtype = GRB.BINARY, name ="x"+str(i)) 
              for i in range(len(scores))]
    
    if warm_start_indices:
        for i in range(len(x_vars)):
            x_vars[i].Start = 0
        for i in warm_start_indices:
            x_vars[i].Start = 1

    m.addConstr(sum(x_vars) == final_num_topics , "N constraint")
    m.setObjective(sum([scores[i]*x_vars[i] for i in range(len(x_vars))]),
                   GRB.MAXIMIZE)
    m.Params.MIPGap = MIP_gap
    m.Params.TimeLimit = time_limit
    
    return m, x_vars

def mdkp(topics, scores, final_num_topics, epsilon, warm_start_indices=None, 
         MIP_gap=0.05, time_limit=3600, **kwargs):
    
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
    warm_start_indices : list of int or None
        indices for starting solution
    MIP_gap : float
        Termination criteria for solution search
        Set 0 to find optimal solution (might be slower)
    time_limit : int
        Time (seconds) before stopping search for solution 

    Returns
    -------
    1-d numpy boolean array
        indicating which topics are chosen
    """
    topics = [set(t) for t in topics]

    words = defaultdict(set)
    for i in range(len(topics)):
        for j in range(i+1, len(topics)):
            for word in topics[i].intersection(topics[j]):
                words[word].add(i)
                words[word].add(j)

    m, x_vars = create_basic_mip(scores, final_num_topics, warm_start_indices, 
                                 MIP_gap, time_limit)

    w_vars = []
    for w in words:
        w_vars.append(m.addVar(vtype = GRB.BINARY, name = w))
#         m.addConstr(w_vars[-1] == gp.max_([x_vars[i] for i in words[w]]))
        m.addConstr(w_vars[-1] <= sum([x_vars[i] for i in words[w]]))
    
    m.addConstr(sum(w_vars) >= epsilon)

    m.optimize()

    return np.array([x_var.X for x_var in x_vars], dtype=bool)
    
def mwbis(topics, scores, final_num_topics, epsilon, warm_start_indices=None, 
          MIP_gap=0.05, time_limit=3600, **kwargs):
    
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
    warm_start_indices : list of int or None
        indices for starting solution
    MIP_gap : float
        Termination criteria for solution search
        Set 0 to find optimal solution (might be slower)
    time_limit : int
        Time (seconds) before stopping search for solution 

    Returns
    -------
    1-d numpy boolean array
        indicating which topics are chosen
    """
        
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
    
    m, x_vars = create_basic_mip(scores, final_num_topics, 
                                 warm_start_indices, MIP_gap, time_limit)

    for w in words:
        m.addConstr(sum([x_vars[i] for i in words[w]]) <= epsilon + 1)
    for i,j in edges:
        m.addConstr(x_vars[i] + x_vars[j] <= 1)

    m.optimize()
    
    return np.array([x_var.X for x_var in x_vars], dtype=bool)