import logging
import os

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
# model
bert_base_model = "../BERT/bert-base-uncased.tar.gz"
bert_base_vocab = "../BERT/bert-base-uncased-vocab.txt"
bert_large_model = "../BERT/bert-large-uncased.tar.gz"
bert_large_vocab = "../BERT/bert-large-uncased-vocab.txt"

train_file = '../../ms-marco/dp0.7/train-yesno-cb-dp70.json'
dev_file = '../../ms-marco/dp0.7/dev-yesno-cb-dp70.json'

task_name = 'marco-cb-dp0.7'
reader_name = 'cb-marco'

bert_name = 'hie'

num_train_epochs = 3.0
learning_rate = 3e-5
evidence_lambda = 0.8
weight_threshold = 0.7
label_threshold = 0.75

root_dir = 'experiments/marco-cb-dp0.7/single-view/v1.0/'

# view0_sentence_id_file = 'experiments/marco-cb-dp0.7/co-training/v2.0_1000/recurrent0_view1/sentence_id_file_recurrent1.json.merge'
view1_sentence_id_file = 'experiments/marco-cb-dp0.7/co-training/v2.0_1000/recurrent0_view0/sentence_id_file_recurrent1.json.merge'

cmd = f"python main_0.6.2.py \
                --bert_model bert-base-uncased \
                --vocab_file {bert_base_vocab} \
                --model_file {bert_base_model} \
                --train_file {train_file} \
                --predict_file {dev_file} \
                --max_seq_length 480 --max_query_length 50 \
                --do_train --do_predict --train_batch_size 8 --predict_batch_size 8 --max_answer_length 15 \
                --num_train_epochs {num_train_epochs} \
                --learning_rate {learning_rate} \
                --max_ctx 3 \
                --bert_name {bert_name} \
                --task_name {task_name} \
                --reader_name {reader_name} \
                --evidence_lambda {evidence_lambda} \
                --do_label \
                --weight_threshold {weight_threshold} \
                --only_correct \
                --label_threshold {label_threshold} " \
    f"--output_dir {root_dir} --predict_dir {root_dir} --sentence_id_files {view1_sentence_id_file} "

logger.info(cmd)
os.system(cmd)

