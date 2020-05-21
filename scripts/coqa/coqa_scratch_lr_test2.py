"""
Train on QuAC from scratch.
Bert + MLP
Bert + hierarchical
Bert + hierarchical + sentence supervision
"""

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
# task setting
# task_name: {dataset}-{simple/stage/pretrain}
task_name = 'coqa'
reader_name = 'coqa'

# Bert + MLP
bert_name = 'mlp'

output_dir = f"experiments/pretrained-bert0.6.2/coqa-mlp/lr_test/"

cmd = f"python main_0.6.2.py \
            --bert_model bert-base-uncased \
            --vocab_file {bert_base_vocab} \
            --model_file {bert_base_model} \
            --output_dir {output_dir} \
            --predict_dir {output_dir}remove_passage/ \
            --train_file ../BERT/max_f1/coqa-train-v1.0.json \
            --predict_file ../BERT/max_f1/coqa-dev-v1.0.json \
            --max_seq_length 512 --max_query_length 385 \
            --do_predict --train_batch_size 8 --predict_batch_size 8 --max_answer_length 15 \
            --learning_rate 3e-5 \
            --num_train_epochs 3.0 \
            --max_ctx 2 \
            --bert_name {bert_name} \
            --task_name {task_name} \
            --reader_name {reader_name} " \
    f"--remove_passage "

print(cmd)
os.system(cmd)

# Bert + hierarchical
bert_name = 'hie'

output_dir = f"experiments/pretrained-bert0.6.2/coqa-hie/lr_test/"

cmd = f"python main_0.6.2.py \
            --bert_model bert-base-uncased \
            --vocab_file {bert_base_vocab} \
            --model_file {bert_base_model} \
            --output_dir {output_dir} \
            --predict_dir {output_dir} \
            --train_file ../BERT/max_f1/coqa-train-v1.0.json \
            --predict_file ../BERT/max_f1/coqa-dev-v1.0.json \
            --max_seq_length 512 --max_query_length 385 \
            --do_predict --train_batch_size 6 --predict_batch_size 6 --max_answer_length 15 \
            --learning_rate 2e-5 \
            --num_train_epochs 3.0 \
            --max_ctx 2 \
            --bert_name {bert_name} \
            --task_name {task_name} \
            --reader_name {reader_name} " \
    f"--evidence_lambda 0.0 " \
    f"--remove_dict /home/jiaofangkai/CoQA-Challenge/BERT/max_f1/remove_dict.json " \
    f"--remove_evidence "

print(cmd)
os.system(cmd)

bert_name = 'hie'

output_dir = f"experiments/pretrained-bert0.6.2/coqa-hie-super/lr_test/"

cmd = f"python main_0.6.2.py \
            --bert_model bert-base-uncased \
            --vocab_file {bert_base_vocab} \
            --model_file {bert_base_model} \
            --output_dir {output_dir} \
            --predict_dir {output_dir} \
            --train_file ../BERT/max_f1/coqa-train-v1.0.json \
            --predict_file ../BERT/max_f1/coqa-dev-v1.0.json \
            --max_seq_length 512 --max_query_length 385 \
            --do_predict --train_batch_size 6 --predict_batch_size 6 --max_answer_length 15 \
            --num_train_epochs 4.0 \
            --learning_rate 4e-5 \
            --max_ctx 2 \
            --bert_name {bert_name} \
            --task_name {task_name} \
            --reader_name {reader_name} " \
    f"--evidence_lambda 0.8 " \
    f"--remove_dict /home/jiaofangkai/CoQA-Challenge/BERT/max_f1/remove_dict.json " \
    f"--remove_evidence "

print(cmd)
os.system(cmd)

