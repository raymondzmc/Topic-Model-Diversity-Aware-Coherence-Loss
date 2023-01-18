conda activate ctm

python run_topic_models.py --text_file resources/20news_unprep.txt --bow_file resources/20news_prep.txt \
--model_type zeroshot --device 3 --num_seeds 5 --use_npmi_loss --use_diversity_loss --weight_lambda 100 --weight_alpha 0.9 | tee 20news_zeroshot_npmi_diversity_scores_0.9.out