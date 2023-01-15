conda activate ctm

python run_topic_models.py --text_file contextualized_topic_models/data/gnews/GoogleNews.txt --bow_file contextualized_topic_models/data/gnews/GoogleNews.txt \
--model_type zeroshot --device 2 --use_mdkp | tee gnews_zeroshot_scores.out

python run_topic_models.py --text_file contextualized_topic_models/data/gnews/GoogleNews.txt --bow_file contextualized_topic_models/data/gnews/GoogleNews.txt \
--model_type zeroshot --device 2 --use_npmi_loss --weight_lambda 100 --use_mdkp | tee gnews_zeroshot_npmi_scores.out