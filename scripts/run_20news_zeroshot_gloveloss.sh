conda activate ctm

python run_topic_models.py --text_file resources/20news_unprep.txt --bow_file resources/20news_prep.txt \
--model_type zeroshot --device 3  --use_glove_loss --weight_lambda 0.001 | tee 20news_zeroshot_glove_scores_0.0005.out

python run_topic_models.py --text_file resources/20news_unprep.txt --bow_file resources/20news_prep.txt \
--model_type zeroshot --device 3  --use_glove_loss --weight_lambda 0.001 | tee 20news_zeroshot_glove_scores_0.001.out

python run_topic_models.py --text_file resources/20news_unprep.txt --bow_file resources/20news_prep.txt \
--model_type zeroshot --device 3  --use_glove_loss --weight_lambda 0.001 | tee 20news_zeroshot_glove_scores_0.005.out

python run_topic_models.py --text_file resources/20news_unprep.txt --bow_file resources/20news_prep.txt \
--model_type zeroshot --device 3  --use_glove_loss --weight_lambda 0.1 | tee 20news_zeroshot_glove_scores_0.1.out





