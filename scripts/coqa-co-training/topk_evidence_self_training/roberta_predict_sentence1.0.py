import logging
import os
import subprocess
import time
import argparse

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, required=True)
args = parser.parse_args()


def run_cmd(command: str):
    logger.info(command)
    subprocess.check_call(command, shell=True)


def wait_for_file(file: str, time_for_writing: int = 1):
    if not os.path.exists(file):
        logger.info(f'Could not find file {file}. Waiting...')
        minute_cnt = 0
        while not os.path.exists(file):
            print(f'The {minute_cnt}th minute...')
            time.sleep(60)
            minute_cnt += 1
        time.sleep(time_for_writing * 60)
        logger.info(f'Find file {file} after waiting for {minute_cnt} minutes')


# model
roberta_large_model_dir = "/home/jiaofangkai/roberta-large"

train_file = '../BERT/max_f1/coqa-train-v1.0.json'
dev_file = '../BERT/max_f1/coqa-dev-v1.0.json'

task_name = 'coqa-top-k-roberta'
reader_name = 'coqa-top-k-roberta'
bert_name = 'hie-topk-roberta'

recurrent_times = 10

sentence_id_file = None

top_k = 1500
root_dir = args.root_dir

f_handler = logging.FileHandler(os.path.join(root_dir, f'output.log'))
f_handler.setLevel(logging.INFO)
f_handler.setFormatter(logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                                         datefmt='%m/%d/%Y %H:%M:%S'))
logger.addHandler(f_handler)

logger.info('Self-training parameters:')
logger.info(f'k: {top_k}')
logger.info(f'recurrent_times: {recurrent_times}')

for i in range(recurrent_times):
    logger.info(f'Running at the {i}th times...')

    if i == 0:
        evidence_lambda = 0.0
        learning_rate = 2e-5
    else:
        learning_rate = 2e-5
        evidence_lambda = 0.8

    output_dir = f"{root_dir}/recurrent{i}/"

    cmd = f"python predict_sentence_main2_0.6.2.py --bert_model roberta-large " \
        f"--vocab_file {roberta_large_model_dir} --model_file {roberta_large_model_dir} " \
        f"--output_dir {output_dir} --predict_dir {output_dir} --train_file {train_file} --predict_file {dev_file} " \
        f"--max_seq_length 512 --max_query_length 385 " \
        f"--train_batch_size 32 --predict_batch_size 1 --max_answer_length 15 --num_train_epochs 4 " \
        f"--learning_rate {learning_rate} --max_ctx 2 --bert_name {bert_name} --task_name {task_name} --reader_name {reader_name} " \
        f"--evidence_lambda {evidence_lambda} "

    # cmd += '--do_train --do_predict '
    cmd += '--do_predict '

    run_cmd(cmd)

    logger.info('=' * 50)
