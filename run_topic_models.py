import os
import random
import argparse

import torch
import numpy as np

from contextualized_topic_models.models.ctm import CombinedTM, ZeroShotTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_file
from contextualized_topic_models.evaluation.measures import TopicDiversity, CoherenceNPMI, CoherenceCV, CoherenceWordEmbeddings, InvertedRBO

import pdb

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


def evaluate(topics, text_for_bow, embeddings_path=None):
    texts = [doc.split() for doc in text_for_bow]

    npmi_metric = CoherenceNPMI(topics, texts)
    npmi_score = npmi_metric.score()
    cv_metric = CoherenceCV(texts=texts, topics=topics)
    cv_score = cv_metric.score()
    we_metric = CoherenceWordEmbeddings(topics=topics)
    we_score = we_metric.score()
    irbo_metric = InvertedRBO(topics=topics)
    irbo_score = irbo_metric.score()
    td_metric = TopicDiversity(topics=topics)
    td_score = td_metric.score(topk=10)
    return npmi_score, cv_score, we_score, irbo_score, td_score



def main(args):
    text_file = os.path.join('resources', '20news_unprep.txt')
    bow_file = os.path.join('resources', '20news_prep.txt')
    text_for_contextual, text_for_bow = load_dataset(text_file, bow_file)

    qt = TopicModelDataPreparation("all-mpnet-base-v2")

    training_dataset = qt.fit(text_for_contextual=text_for_contextual, text_for_bow=text_for_bow)

    for num_topics in [25, 50, 75, 100, 150]:
        npmi_scores = []
        cv_scores = []
        we_scores = []
        irbo_scores = []
        td_scores = []

        for seed in range(30):
            set_random_seed(seed)
            ctm = CombinedTM(bow_size=len(qt.vocab), contextual_size=768, n_components=num_topics)

            ctm.fit(training_dataset)
            topics = [v for k, v in ctm.get_topics(10).items()]
            scores = evaluate(topics, text_for_bow, embeddings_path=None)

            npmi_scores.append(scores[0])
            cv_scores.append(scores[1])
            we_scores.append(scores[2])
            irbo_scores.append(scores[3])
            td_scores.append(scores[4])

        print(f"[{num_topics}-Topics] NPMI: {np.mean(npmi_scores)}, CV: {np.mean(npmi_scores)}, WE: {np.mean(we_scores)}, I-RBO: {np.mean(irbo_scores)}, TD: {np.mean(td_scores)}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", help="Dataset name", default="20news", choices=["20news", 'wiki'])
    args = parser.parse_args()
    # parser.add_argument("--text_file", help="Dataset name", default="20news", choices=["20news", 'wiki'])
    main(args)
