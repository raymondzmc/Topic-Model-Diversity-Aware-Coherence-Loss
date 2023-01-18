conda activate ctm

# python run_topic_models.py --text_file resources/20news_unprep.txt --bow_file resources/20news_prep.txt \
# --model_type combined --device 0  | tee 20news_combined_scores.out

# python run_topic_models.py --text_file resources/20news_unprep.txt --bow_file resources/20news_prep.txt \
# --model_type combined --device 0 --use_npmi_loss --weight_lambda 100 | tee 20news_combined_npmi_scores.out

python run_topic_models.py --text_file resources/20news_unprep.txt --bow_file resources/20news_prep.txt \
--model_type combined --device 2 --use_npmi_loss --weight_lambda 100 --use_diversity_loss --weight_alpha 0.7 \
| tee 20news_combined_npmi_diversity_scores.out