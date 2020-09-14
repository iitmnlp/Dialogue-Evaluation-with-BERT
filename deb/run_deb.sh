export STORAGE_BUCKET=gs://ana_reddit_bucket
export BERT_BASE_DIR=gs://cloud-tpu-checkpoints/bert/uncased_L-12_H-768_A-12

python3 deb.py --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=${STORAGE_BUCKET}/deb_trained_on_rand_and_adv_neg/model.ckpt-102870 --num_warmup_steps=100 --input_file=${STORAGE_BUCKET}/daily/DDpp_hard_neg_test.tfrecord --output_dir=${STORAGE_BUCKET}/daily/output --do_train=False --max_eval_steps=475 --use_tpu=True --tpu_name=tpu1 --do_eval=True --max_predictions_per_seq=0 

#python3 deb.py --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=${STORAGE_BUCKET}/deb_trained_on_rand_neg/model.ckpt-3214 --num_warmup_steps=100 --input_file=${STORAGE_BUCKET}/daily/DDpp_hard_neg_test.tfrecord --output_dir=${STORAGE_BUCKET}/daily/output --do_train=False --max_eval_steps=475 --use_tpu=True --tpu_name=tpu1 --do_eval=True --max_predictions_per_seq=0 
