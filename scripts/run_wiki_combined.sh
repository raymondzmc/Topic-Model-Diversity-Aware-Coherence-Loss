conda activate ctm

python run_topic_models.py --text_file resources/dbpedia_sample_abstract_20k_unprep.txt --bow_file resources/dbpedia_sample_abstract_20k_prep.txt \
--model_type combined --device 3 --use_mdkp | tee wiki_combined_scores.out

python run_topic_models.py --text_file resources/dbpedia_sample_abstract_20k_unprep.txt --bow_file resources/dbpedia_sample_abstract_20k_prep.txt \
--model_type combined --device 3 --use_npmi_loss --weight_lambda 100 --use_mdkp | tee wiki_combined_npmi_scores.out