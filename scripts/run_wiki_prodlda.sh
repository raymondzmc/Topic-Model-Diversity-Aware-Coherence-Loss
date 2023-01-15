conda activate ctm

python run_topic_models.py --text_file resources/dbpedia_sample_abstract_20k_unprep.txt --bow_file resources/dbpedia_sample_abstract_20k_prep.txt \
--model_type prodlda --device 2 --use_mdkp | tee wiki_prodlda_scores.out

python run_topic_models.py --text_file resources/dbpedia_sample_abstract_20k_unprep.txt --bow_file resources/dbpedia_sample_abstract_20k_prep.txt \
--model_type prodlda --device 2 --use_npmi_loss --weight_lambda 100 --use_mdkp | tee wiki_prodlda_npmi_scores.out