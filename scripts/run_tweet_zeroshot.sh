conda activate ctm

python run_topic_models.py --text_file resources/STTM/Tweet.txt --bow_file resources/STTM/Tweet.txt \
--model_type combined --device 1 | tee tweet_combined_scores.out

python run_topic_models.py --text_file resources/STTM/Tweet.txt --bow_file resources/STTM/Tweet.txt \
--model_type combined --device 1 --use_npmi_loss --weight_lambda 100 | tee tweet_combined_npmi_scores.out