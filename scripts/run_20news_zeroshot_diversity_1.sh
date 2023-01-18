conda activate ctm

# python run_topic_models.py --text_file resources/20news_unprep.txt --bow_file resources/20news_prep.txt \
# --model_type zeroshot --device 1 | tee 20news_zeroshot_scores.out

python run_topic_models.py --text_file resources/20news_unprep.txt --bow_file resources/20news_prep.txt \
--model_type zeroshot --device 3  --use_npmi_loss --use_diversity_loss --weight_lambda 100 --weight_alpha 1 | tee 20news_zeroshot_npmi_diversity_scores_1.out