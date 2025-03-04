conda activate ctm

# python run_topic_models.py --text_file resources/dbpedia_sample_abstract_20k_unprep.txt --bow_file resources/dbpedia_sample_abstract_20k_prep.txt \
# --model_type prodlda --device 3 --use_mdkp | tee wiki_prodlda_scores.out

# python run_topic_models.py --text_file resources/dbpedia_sample_abstract_20k_unprep.txt --bow_file resources/dbpedia_sample_abstract_20k_prep.txt \
# --model_type prodlda --device 3 --use_npmi_loss --weight_lambda 100 --use_mdkp | tee wiki_prodlda_npmi_scores.out

python run_topic_models.py --text_file resources/dbpedia_sample_abstract_20k_unprep.txt --bow_file resources/dbpedia_sample_abstract_20k_prep.txt \
--model_type prodlda --device 1 --use_npmi_loss --weight_lambda 100 --use_diversity_loss --weight_alpha 0.7 \
| tee wiki_prodlda_npmi_diversity_scores.out