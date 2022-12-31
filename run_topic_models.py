import os
import random
import argparse
from os.path import join as pjoin

import torch
import numpy as np
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_distances

from contextualized_topic_models.models.ctm import CombinedTM, ZeroShotTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_file
from contextualized_topic_models.evaluation.measures import TopicDiversity, CoherenceNPMI, CoherenceCV, CoherenceWordEmbeddings, InvertedRBO
from contextualized_topic_models.utils.visualize import save_word_dist_plot

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



def main(args):
    text_for_contextual, text_for_bow = load_dataset(args.text_file, args.bow_file)
    qt = TopicModelDataPreparation("all-mpnet-base-v2", device=args.device)

    dataset_name = os.path.basename(args.text_file).split('_')[0]
    model_type = 'combined' if args.concat_bow else 'zeroshot'

    if args.cache_dataset:
        cache_file = pjoin(args.cache_path, f"{dataset_name}-{args.model_name}-{model_type}.pt")
        if os.path.isfile(cache_file):
            training_dataset = torch.load(cache_file)
        else:
            training_dataset = qt.fit(text_for_contextual=text_for_contextual, text_for_bow=text_for_bow)
            torch.save(training_dataset, cache_file)
    else:
        training_dataset = qt.fit(text_for_contextual=text_for_contextual, text_for_bow=text_for_bow)

    if args.use_dist_loss:
        model_type += '-distloss'
        wv = api.load('word2vec-google-news-300')
        word_vectors = np.zeros((len(training_dataset.idx2token), wv.vector_size))
        missing_indices = []
        for idx, token in training_dataset.idx2token.items():
            if wv.has_index_for(token):
                word_vectors[idx] = wv.get_vector(token)
            else:
                missing_indices.append(idx)
        
        # Use the mean vector for OOV tokens
        vocab_mask = np.ones(len(word_vectors), dtype=bool)
        vocab_mask[missing_indices] = False
        # word_vectors[missing_mask] = word_vectors[~missing_mask].mean(axis=0)
    
        # Compute normalized distance matrix
        dist_matrix = cosine_distances(word_vectors)
        for row_idx in range(len(dist_matrix)):
            row = dist_matrix[row_idx]
            dist_matrix[row_idx] = (row - row.min()) / (row.max() - row.min())
    else:
        dist_matrix = None
    
    
    for num_topics in [25, 50, 75, 100, 150]:
        npmi_scores = []
        # cv_scores = []
        we_scores = []
        irbo_scores = []
        td_scores = []

        for seed in range(args.num_seeds):
            set_random_seed(seed)

            # Concatenate BoW input with embeddings in CombinedTM (https://aclanthology.org/2021.acl-short.96.pdf)
            if args.concat_bow:
                ctm = CombinedTM(
                    bow_size=len(training_dataset.idx2token),
                    contextual_size=768,
                    n_components=num_topics,
                    device=args.device,
                    use_dist_loss=args.use_dist_loss,
                    dist_matrix=dist_matrix,
                    vocab_mask=vocab_mask,
                    loss_weights={"lambda": args.weight_lambda, "beta": 1},
                )

            # Use only contextualized embeddings in ZeroShotTM (https://aclanthology.org/2021.eacl-main.143.pdf)
            else:
                ctm = ZeroShotTM(
                    bow_size=len(training_dataset.idx2token),
                    contextual_size=768,
                    n_components=num_topics,
                    device=args.device,
                    use_dist_loss=args.use_dist_loss,
                    dist_matrix=dist_matrix,
                    vocab_mask=vocab_mask,
                    loss_weights={"lambda": args.weight_lambda, "beta": 1},
                )

            

            ctm.fit(training_dataset)
            topics = [v for k, v in ctm.get_topics(10).items()]
            scores = evaluate(topics, text_for_bow, embeddings_path=None)
            npmi_scores.append(scores[0])
            # cv_scores.append(scores[1])
            we_scores.append(scores[1])
            irbo_scores.append(scores[2])
            td_scores.append(scores[3])

            if args.plot_word_dist:
                for topic_idx, beta in enumerate(ctm.model.beta):
                    save_word_dist_plot(
                        torch.softmax(beta, 0), training_dataset.vocab, 
                        pjoin(args.results_path, f"{dataset_name}-{args.model_name}-{model_type}-{num_topics}topic-lambda{args.weight_lambda}-{topic_idx}.png"),
                        top_n=100)
        
        # print(f"[{num_topics}-Topics] NPMI: {np.mean(npmi_scores)}, CV: {np.mean(npmi_scores)}, WE: {np.mean(we_scores)}, I-RBO: {np.mean(irbo_scores)}, TD: {np.mean(td_scores)}")
        print(f"[{num_topics}-Topics] NPMI: {np.mean(npmi_scores)}, CV: {np.mean(npmi_scores)}, WE: {np.mean(we_scores)}, I-RBO: {np.mean(irbo_scores)}, TD: {np.mean(td_scores)}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--text_file", help="Unprocessed text file path", default=pjoin('resources', '20news_unprep.txt'), type=str)
    parser.add_argument("--bow_file", help="Processed bag-of-words file path", default=pjoin('resources', '20news_prep.txt'), type=str)
    # parser.add_argument("--result", help="Processed bag-of-words file path", default=pjoin('resources', '20news_prep.txt'), type=str)

    parser.add_argument("--model_name", help="Name for pre-trained model for generating contextualized embeddings", default="all-mpnet-base-v2", type=str)
    parser.add_argument("--concat_bow", help="Whether to concatenate BoW representation with contextualized embeddings", action='store_true')
    parser.add_argument("--device", help="Device for model training/inference", default=None)

    # Experiments
    parser.add_argument("--cache_dataset", help="Cache dataset object before training",  action='store_true')
    parser.add_argument("--cache_path", help="Path for caching", type=str, default='.cache')
    parser.add_argument("--num_seeds", help="Number of random seeds for each topic number", type=int, default=30)
    parser.add_argument("--results_path", help="Path for saving results", type=str, default='results')
    parser.add_argument("--plot_word_dist", help="Visualize topic-word distribution over vocab",  action='store_true')

    # CTM hyperparameters (default to implementation from orginal paper)
    parser.add_argument("--hidden_sizes", help="Hidden size", default=(100, 100), type=tuple)
    parser.add_argument("--activation", help="Activation function", default='softplus', type=str, choices=['softplus', 'relu'])
    parser.add_argument("--dropout", help="Dropout rate", default=0.2, type=float)
    parser.add_argument("--batch_size", help="Batch size for training", default=100, type=int)
    parser.add_argument("--lr", help="Learning rate", default=2e-3, type=float)
    parser.add_argument("--momentum", help="Momentum for optimizer", default=0.99, type=float)
    parser.add_argument("--num_epochs", help="Number of epochs for training", default=100, type=int)

    parser.add_argument("--use_dist_loss", help="Use embedding distance loss", action='store_true')
    parser.add_argument("--weight_lambda", help="Weight for distance loss", type=float, default=10)
    parser.add_argument("--divergence_loss", help="Use topic divergence loss", action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.cache_path):
        os.makedirs(args.cache_path)
    
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    try:
        args.device = torch.device(f'cuda:{int(args.device)}')
    except:
        args.device = 'cpu'
    main(args)
