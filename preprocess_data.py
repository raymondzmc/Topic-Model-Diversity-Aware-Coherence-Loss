import os
import pdb
import argparse
import string
from gensim.utils import deaccent
import warnings
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_file", help="Path to input text file", required=True)
    parser.add_argument("--vocab_file", help="Path to vocab file", default=None)
    parser.add_argument("--output_file", help="Path to output BoW file", required=True)
    args = parser.parse_args()


    with open(args.vocab_file, 'r') as f:
        vocab = set([w.strip() for w in f.readlines()])

    with open(args.text_file, 'r') as f:
        documents = []
        for line in f:
            documents.append(line.strip())
    
    preprocessed_docs_tmp = [deaccent(doc.lower()) for doc in documents]
    preprocessed_docs_tmp = [doc.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
                                for doc in preprocessed_docs_tmp]
    preprocessed_docs_tmp = [' '.join([w for w in doc.split() if len(w) > 0])
                                for doc in preprocessed_docs_tmp]
    preprocessed_docs_tmp = [' '.join([w for w in doc.split() if w in vocab])
                                 for doc in preprocessed_docs_tmp]
    
    word_set = set()
    preprocessed_docs, unpreprocessed_docs, retained_indices = [], [], []
    for i, doc in enumerate(preprocessed_docs_tmp):
        if len(doc) > 1:
            preprocessed_docs.append(doc)
            unpreprocessed_docs.append(documents[i])
            retained_indices.append(i)
            word_set.update(doc.split())

    if len(retained_indices) != len(documents):
        pdb.set_trace()
    
    pdb.set_trace()
    with open(args.output_file, 'w+') as f:
        for line in preprocessed_docs:
            f.write(line + '\n')
    