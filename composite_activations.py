import os
import pickle
import numpy as np
import pandas as pd
from algo.normal import calculate_compositions
from mlxtend.frequent_patterns import apriori

from octis.evaluation_metrics.coherence_metrics import Coherence
from gensim.models import CoherenceModel

def composite_activations(beta, data):
	mat = np.sort(data, axis=1).mean(axis=0)
	assert len(mat) == K

	# Find a suitable threshold kappa hyper-parameter (currently using 5th largest mean activation)
	threshold = mat[-5]
	print('threshold', threshold)

	reduced_data = np.zeros_like(beta)
	for i,j in np.argwhere(beta > threshold):
	    reduced_data[i,j] = 1

	reduced_data = pd.DataFrame(reduced_data)

	min_s = 0.01
	frequent_itemsets = apriori(reduced_data, 
	                            min_support = min_s, 
	                            max_len = 5, 
	                            use_colnames = True,
	                           verbose = 1)

	topic_combinations = [list(a) for a in frequent_itemsets['itemsets']]


	output, topic_combinations = calculate_compositions(beta, topic_combinations, add_pairs=True)

	coherence = Coherence(texts=dataset.get_corpus(), measure = 'c_npmi')
	c = CoherenceModel(topics=topics, texts=coherence._texts, dictionary=coherence._dictionary,
	                                      coherence='c_npmi', processes=4, topn=coherence.topk)
	total_scores = c.get_coherence_per_topic()

	choices = mdkp(topics, total_scores, K, 0.935*K*10, range(K), MIP_gap=0.01, time_limit=3600)

	for i in range(len(choices)):
	    if choices[i]:
	        print("{0: .3f}".format(total_scores[i]), " ".join(topics[i]))
	        if i < K:
	            print('\t> original component topic')
	            continue
	        for c in topic_combinations[i]:
	            print("\t>{0: .3f}".format(total_scores[c]), " ".join(topics[c]))