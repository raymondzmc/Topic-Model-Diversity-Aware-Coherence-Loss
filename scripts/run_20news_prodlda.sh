conda activate ctm

python run_topic_models.py --text_file resources/20news_unprep.txt --bow_file resources/20news_prep.txt \
--model_type prodlda --device 3 --use_mdkp | tee 20news_prodlda_scores.out

python run_topic_models.py --text_file resources/20news_unprep.txt --bow_file resources/20news_prep.txt \
--model_type prodlda --device 3  --use_npmi_loss --weight_lambda 100 --use_mdkp | tee 20news_prodlda_npmi_scores.out