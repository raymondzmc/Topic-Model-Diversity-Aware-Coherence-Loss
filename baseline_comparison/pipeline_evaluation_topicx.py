

#python3 pipeline_evaluation_topicx.py --name_dataset '20news'  --number_of_topic_models 10 --list_number_of_topics "25,50,75,100,150" 

# %%
import sys
sys.path.append('topicx/')

from baselines.cetopictm import CETopicTM
from utils import prepare_dataset
import baseline_utils
from sklearn.feature_extraction.text import CountVectorizer
from octis.dataset.dataset import Dataset
import random 
import pickle

import argparse




parser = argparse.ArgumentParser()
parser.add_argument("--name_dataset", help="Name of dataset", choices=['20news', 'dbpedia', 'google_news'])
parser.add_argument("--number_of_topic_models", help="Number of iterations over a topic model", type=int,  default=1)
parser.add_argument('--list_number_of_topics', help='List of number of topics, e.g, "5,15,50"', type=str)


# %%
args = parser.parse_args()

dataset_name = args.name_dataset
number_topic_models = args.number_of_topic_models

list_number_of_topics = [int(x) for x in args.list_number_of_topics.split(',')]

sbert_model_name = 'all-mpnet-base-v2'
word2vec_path = '/mnt/datasets/VIST/data/commonsense_ctm_data/GoogleNews-vectors-negative300.bin.gz'


# %%
dataset = Dataset()

if dataset_name == '20news':
    dataset.load_custom_dataset_from_folder("resources_octis/20news")
if dataset_name == 'dbpedia':
    dataset.load_custom_dataset_from_folder("resources_octis/dbpedia")
if dataset_name =='google_news':
    dataset.load_custom_dataset_from_folder("resources_octis/google_news")

# %%
texts = dataset.get_corpus()
texts = [' '.join(text) for text in texts]

# %%

key_name = 'results/topicx_'+dataset_name+'_NTM_'+str(number_topic_models)+'_ntopics_'+str(list_number_of_topics)+'_embedding_'+sbert_model_name+'.pkl'
print('key_name', key_name)

# %%

from datetime import datetime
start_time = datetime.now()

final_results = []
for current_number_of_topics in list_number_of_topics:
    for i in range(number_topic_models):
        random_seed_number = random.randint(0, 1000)
        print('Random seed number: {}'.format(random_seed_number))


        tm = CETopicTM(dataset=dataset, 
                    topic_model='cetopic', 
                    num_topics=current_number_of_topics, 
                    dim_size=50,#In the paper the authors say 'we reduce the dimensionality of sentence embedding to 50 usign UMAP' #default 5 
                    word_select_method='tfidf_idfi', #best word selection method according to their paper
                    embedding='/mnt/datasets/SBERT/'+sbert_model_name,
                    seed=random_seed_number)  #sentence-transformers/all-mpnet-base-v2', #embedding='princeton-nlp/unsup-simcse-bert-base-uncased',  #Default in Raymond's evaluation is: all-mpnet-base-v2 #TODO: I think we must use a different embedding)
        
        tm.train()

        #td_score, cv_score, npmi_score = tm.evaluate()
        #print(f'td: {td_score} npmi: {npmi_score} cv: {cv_score}')

        topics = tm.get_topics()
        print(f'Topics: {topics}')

        formatted_topics_list = []

        for key, value in topics.items():
            #new_list.append(topic.split())
            list_current_topic = []
            for keyword, score in value:
                list_current_topic.append(keyword)
            #print(list_current_topic)
            formatted_topics_list.append(list_current_topic)

        npmi_score, we_score, irbo_score, td_score = baseline_utils.evaluate(formatted_topics_list, texts, embeddings_path=word2vec_path)
        dict_results = {'npmi_score': npmi_score, 'we_score': we_score, 'irbo_score': irbo_score, 'td_score': td_score}
        print(dict_results)
        final_results.append(dict_results)

        output = open(key_name, 'wb')
        pickle.dump(final_results, output)
        output.close()
        end_time  =datetime.now()
        print('Duration: {}'.format(end_time - start_time))

output = open(key_name, 'wb')
pickle.dump(final_results, output)
output.close()    







