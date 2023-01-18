conda activate ctm

python run_topic_models.py --text_file resources/20news_unprep.txt --bow_file resources/20news_prep.txt \
--model_type zeroshot --device 1 --num_seeds 5 --use_npmi_loss --weight_lambda 100 --use_diversity_loss --weight_alpha 0.7 \
 | tee 20news_zeroshot_npmi_diversity_scores_0.7.out