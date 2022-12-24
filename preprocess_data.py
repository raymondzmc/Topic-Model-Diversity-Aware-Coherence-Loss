import os
import pdb
import argparse
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_file", help="Path to input text file", required=True)
    parser.add_argument("--vocab_file", help="Path to vocab file", default=None)
    parser.add_argument("--output_file", help="Path to output BoW file", required=True)
    args = parser.parse_args()


    with open(args.vocab_file, 'r') as f:
        vocab = set([w.strip() for w in f.readlines()])


    with open(args.text_file, 'r') as f:
        for line in f:
            pdb.set_trace()