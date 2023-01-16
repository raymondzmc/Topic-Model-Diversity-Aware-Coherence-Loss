conda activate ctm

python run_topic_models.py --text_file resources/20news_unprep.txt --bow_file resources/20news_prep.txt \
--model_type zeroshot --device 3  --use_glove_loss --weight_lambda 0.001 | tee 20news_zeroshot_glove_scores.out

python run_topic_models.py --text_file resources/20news_unprep.txt --bow_file resources/20news_prep.txt \
--model_type zeroshot --device 3  --use_glove_loss --weight_lambda 0.1 | tee 20news_zeroshot_glove_scores.out

python run_topic_models.py --text_file resources/20news_unprep.txt --bow_file resources/20news_prep.txt \
--model_type zeroshot --device 3  --use_glove_loss --weight_lambda 1 | tee 20news_zeroshot_glove_scores.out

python run_topic_models.py --text_file resources/20news_unprep.txt --bow_file resources/20news_prep.txt \
--model_type zeroshot --device 3  --use_glove_loss --weight_lambda 5 | tee 20news_zeroshot_glove_scores.out





