"""
Train on QuAC from scratch.
Bert + MLP
Bert + hierarchical
Bert + hierarchical + sentence supervision
"""

import logging
import os
import subprocess

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
# model
bert_base_model = "../BERT/bert-base-uncased.tar.gz"
bert_base_vocab = "../BERT/bert-base-uncased-vocab.txt"
bert_large_model = "../BERT/bert-large-uncased.tar.gz"
bert_large_vocab = "../BERT/bert-large-uncased-vocab.txt"
# task setting
# task_name: {dataset}-{simple/stage/pretrain}
task_name = 'coqa'
reader_name = 'coqa'

"""
Use k evidence sentences with higher attention weights.
"""

# Bert + hierarchical
bert_name = 'twoview'

output_dir = 'experiments/coqa/co-training/v7.0_1000/recurrent5_view0'
predict_dir = f"experiments/only_evidence/coqa-co-training/v7.0-view0-k=2/"

cmd = f"python main_0.6.2.py \
            --bert_model bert-base-uncased \
            --vocab_file {bert_base_vocab} \
            --model_file {bert_base_model} \
            --output_dir {output_dir} \
            --predict_dir {predict_dir} \
            --train_file ../BERT/max_f1/coqa-train-v1.0.json \
            --predict_file ../BERT/max_f1/coqa-dev-v1.0.json \
            --max_seq_length 512 --max_query_length 385 \
            --do_predict --train_batch_size 6 --predict_batch_size 6 --max_answer_length 15 \
            --learning_rate 2e-5 \
            --num_train_epochs 3.0 \
            --max_ctx 2 \
            --bert_name {bert_name} \
            --task_name {task_name} \
            --reader_name {reader_name} --view_id 0 " \
    f"--evidence_lambda 0.0"

print(cmd)
subprocess.check_call(cmd, shell=True)

output_dir = 'experiments/coqa/co-training/v7.0_1000/recurrent5_view1'
predict_dir = f"experiments/only_evidence/coqa-co-training/v7.0-view1-k=2"

cmd = f"python main_0.6.2.py \
            --bert_model bert-base-uncased \
            --vocab_file {bert_base_vocab} \
            --model_file {bert_base_model} \
            --output_dir {output_dir} \
            --predict_dir {predict_dir} \
            --train_file ../BERT/max_f1/coqa-train-v1.0.json \
            --predict_file ../BERT/max_f1/coqa-dev-v1.0.json \
            --max_seq_length 512 --max_query_length 385 \
            --do_predict --train_batch_size 6 --predict_batch_size 6 --max_answer_length 15 \
            --num_train_epochs 4.0 \
            --learning_rate 4e-5 \
            --max_ctx 2 \
            --bert_name {bert_name} \
            --task_name {task_name} \
            --reader_name {reader_name} --view_id 1 " \
    f"--evidence_lambda 0.8"

print(cmd)
subprocess.check_call(cmd, shell=True)
