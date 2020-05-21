import logging
import os
import subprocess

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def run_cmd(command: str):
    logger.info(command)
    subprocess.check_call(command, shell=True)


# model
bert_base_model = "../BERT/bert-base-uncased.tar.gz"
bert_base_vocab = "../BERT/bert-base-uncased-vocab.txt"
bert_large_model = "../BERT/bert-large-uncased.tar.gz"
bert_large_vocab = "../BERT/bert-large-uncased-vocab.txt"

train_file = '../BERT/max_f1/coqa-train-v1.0.json'
dev_file = '../BERT/max_f1/coqa-dev-v1.0.json'

task_name = 'coqa'
reader_name = 'coqa'

bert_name = 'hie-reinforce'

num_train_epochs = 3.0
learning_rate = 2e-5
evidence_lambda = 0.0
reward_func = 1

output_dir = 'experiments/coqa/reinforce/reinforce-tune/v3.0/'

cmd = f"python main_0.6.2.py --bert_model bert-base-uncased --vocab_file {bert_base_vocab} --model_file {bert_base_model} " \
    f"--train_file {train_file} --predict_file {dev_file} --max_seq_length 512 --max_query_length 385 " \
    f"--do_train --do_predict --train_batch_size 6 --predict_batch_size 6 --max_answer_length 15 " \
    f"--num_train_epochs {num_train_epochs} --learning_rate {learning_rate} --max_ctx 2 " \
    f"--bert_name {bert_name} --task_name {task_name} --reader_name {reader_name} " \
    f"--output_dir {output_dir} --predict_dir {output_dir} " \
    f"--evidence_lambda {evidence_lambda} --sample_steps 10 --reward_func {reward_func} " \
    f"--pretrain experiments/coqa/reinforce/gumbel-pre-train/v2.0/pytorch_model.bin "

run_cmd(cmd)

cmd = f"python predict_sentence_main0.6.2.py --bert_model bert-base-uncased --vocab_file {bert_base_vocab} --model_file {bert_base_model} " \
    f"--train_file {train_file} --predict_file {dev_file} --max_seq_length 512 --max_query_length 385 " \
    f"--do_predict --train_batch_size 6 --predict_batch_size 6 --max_answer_length 15 " \
    f"--num_train_epochs {num_train_epochs} --learning_rate {learning_rate} --max_ctx 2 " \
    f"--bert_name {bert_name} --task_name {task_name} --reader_name {reader_name} " \
    f"--output_dir {output_dir} --predict_dir {output_dir} " \
    f"--evidence_lambda {evidence_lambda} --sample_steps 10 --reward_func {reward_func} "

run_cmd(cmd)
