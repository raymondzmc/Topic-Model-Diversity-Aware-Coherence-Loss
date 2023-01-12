import numpy as np
from functools import reduce
import operator
from itertools import combinations

def heuristic(t, selected, words_cap, topics, limit):
    """Returns True when:
    1) If all words in the query topic does not exceed limit when topic is included
    2) Number of duplicate words does not exceed limit

    Parameters
    ----------
    t : int
        index of query topic
    selected : set of int
        indices of selected topics
    words_cap : dict
        counts of words of selected topics
    topics: list of list of identifier tokens
        Tokens can represent word or vocab identifiers
    limit : int
        Parameter controlling uniqueness of topics. 
        
    Returns
    -------
    bool
        indicating decision to choose
    """
    if t in selected:
        return False
    for w in topics[t]:
        if w in words_cap and words_cap[w] > limit+1:
            return False
    if sum([1 for w in topics[t] if w in words_cap]) > limit:
        return False
    return True

def greedy(topics, scores, final_topic_num, epsilon, max_iteration=10, **kwargs):
    """Greedy algorithm to decide final set of composite topics

    Parameters
    ----------
    topics : list of list of identifier tokens
        Tokens can represent word or vocab identifiers
    scores : list of floats
        Fitness score to be used for decisions
    final_topic_num : int
        (K) Number of chosen topics
    epsilon : int
        Hyper-parameter controlling uniqueness of topics. 
        Set 0 for unique topics.
        Setting higher values might decrease uniqueness.
    max_iteration : int
        Number of further relaxation of epsilon
        Set it to topic length to guarantee a solution

    Returns
    -------
    1-d numpy boolean array
        indicating which topics are chosen
    """
    
    sorted_scores_index = np.argsort(scores)[::-1]
    words_cap = {}
    selected_topics_index = []
    
    for limit in range(epsilon, max_iteration):
        for topic_index in sorted_scores_index:
            
            if heuristic(topic_index, selected_topics_index, words_cap, topics, limit):
                selected_topics_index.append(topic_index)
                for w in topics[topic_index]:
                    if w not in words_cap:
                        words_cap[w] = 0
                    words_cap[w] += 1
                    
            if len(selected_topics_index) == final_topic_num:
                break
    
    selected_topics_boolean = np.zeros(len(scores), dtype=bool)
    selected_topics_boolean[selected_topics_index] = True
    
    return selected_topics_boolean


def count_unique_words(topics):
    """
    Parameters
    ----------
    topics : list of list of identifier tokens
        Tokens can represent word or vocab identifiers 
        
    Returns
    -------
    int
        Number of unique tokens
    """
    return len(reduce(operator.or_, [set(t) for t in topics], set()))

def calculate_compositions(beta, topic_combinations, add_pairs=True):
    """
    Parameters
    ----------
    beta : Matrix of size V x K
        Representing word-topic distribution
    topic_combinations : list of list of topic identifiers
        Each list of topic identifiers represents a composition
    add_pairs : bool, optional
        Include compositional pairs if missing from topic_combinations
        
    Returns
    -------
    int
        Number of unique tokens
    """

    calculations = []
    existing_pairs = set()
    for t in topic_combinations:
        calculations.append(np.sum(beta[:,t], axis=1))
        if len(t) == 2:
            if t[1] > t[0]:
                existing_pairs.add(tuple(t))
            else:
                existing_pairs.add((t[1],t[0]))

    # add missing pair combinations (optional)
    if add_pairs:
        for i,j in combinations(list(range(beta.shape[-1])),2):
            if (i,j) not in existing_pairs:
                topic_combinations.append((i,j))
                calculations.append(np.sum(beta[:,[i,j]], axis=1))
            
    return calculations, topic_combinations
