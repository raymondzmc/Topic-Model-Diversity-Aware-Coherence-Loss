import os
import random
import argparse
import itertools
from os.path import join as pjoin

import torch
import numpy as np
import gensim
import gensim.downloader as api
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm


#from contextualized_topic_models.models.ctm import CombinedTM, ZeroShotTM
#from contextualized_topic_models.models.pytorchavitm.avitm.avitm_model import AVITM_model
#from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_file
from contextualized_topic_models.evaluation.measures import TopicDiversity, CoherenceNPMI, CoherenceCV, CoherenceWordEmbeddings, InvertedRBO
#from contextualized_topic_models.utils.visualize import save_word_dist_plot, save_histogram
#from composite_activations import composite_activations

#import pdb

# Disable tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(0)

def load_dataset(text_file, bow_file):
    input_text = []
    input_bow = []
    vocab = set()

    with open(text_file, 'r') as f:
        for line in f:
            input_text.append(line.rstrip('\n'))

    with open(bow_file, 'r') as f:
        for line in f:
            input_bow.append(line.rstrip('\n'))
            vocab.update(line.split())

    assert len(input_text) == len(input_bow), \
        f"The number of lines in \"{text_file}\" ({len(input_text)}) does not match the number of lines in {bow_file} ({len(input_bow)})!"

    print(f"Successfully read {len(input_text)} documents with a vocab of {len(vocab)}.")
    return input_text, input_bow


def evaluate(topics, texts, embeddings_path=None):
    texts = [doc.split() for doc in texts]

    npmi_metric = CoherenceNPMI(texts=texts, topics=topics)
    npmi_score = npmi_metric.score()
    # cv_metric = CoherenceCV(texts=texts, topics=topics)
    # cv_score = cv_metric.score()
    we_metric = CoherenceWordEmbeddings(topics=topics)
    we_score = we_metric.score()
    irbo_metric = InvertedRBO(topics=topics)
    irbo_score = irbo_metric.score()
    td_metric = TopicDiversity(topics=topics)
    td_score = td_metric.score(topk=10)
    return npmi_score, we_score, irbo_score, td_score

