conda activate ctm

python run_topic_models.py --text_file resources/STTM/Tweet.txt --bow_file resources/STTM/Tweet.txt \
--model_type prodlda --device 0 | tee tweet_prodlda_scores.out

python run_topic_models.py --text_file resources/STTM/Tweet.txt --bow_file resources/STTM/Tweet.txt \
--model_type prodlda --device 0 --use_npmi_loss --weight_lambda 100 | tee tweet_prodlda_npmi_scores.out