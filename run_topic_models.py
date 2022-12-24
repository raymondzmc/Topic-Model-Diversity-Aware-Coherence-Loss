import os
import argparse


from contextualized_topic_models.models.ctm import CombinedTM, ZeroShotTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_file

# Disable tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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


def main(args):
    text_file = os.path.join('resources', '20news_unprep.txt')
    bow_file = os.path.join('resources', '20news_prep.txt')
    text_for_contextual, text_for_bow = load_dataset(text_file, bow_file)

    qt = TopicModelDataPreparation("all-mpnet-base-v2")

    training_dataset = qt.fit(text_for_contextual=text_for_contextual, text_for_bow=text_for_bow)

    ctm = ZeroShotTM(bow_size=len(qt.vocab), contextual_size=768, n_components=50)

    ctm.fit(training_dataset)
    ctm.get_topics(2)
    pdb.set_trace()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", help="Dataset name", default="20news", choices=["20news", 'wiki'])
    args = parser.parse_args()
    # parser.add_argument("--text_file", help="Dataset name", default="20news", choices=["20news", 'wiki'])
    main(args)
