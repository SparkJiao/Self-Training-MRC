import logging
import os
import subprocess
import time

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


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
bert_base_model = "../BERT/bert-base-uncased.tar.gz"
bert_base_vocab = "../BERT/bert-base-uncased-vocab.txt"
bert_large_model = "../BERT/bert-large-uncased.tar.gz"
bert_large_vocab = "../BERT/bert-large-uncased-vocab.txt"

train_file = '../../ms-marco/dp0.7/train-yesno-cb-dp70.json'
dev_file = '../../ms-marco/dp0.7/dev-yesno-cb-dp70.json'

task_name = 'marco-cb-dp0.7-topk'
reader_name = 'cb-marco-top-k'
bert_name = 'hie-topk-32'

evidence_lambda = 0.0
learning_rate = 2e-5

root_dir = f"experiments/marco-cb-dp0.7/topk-rule/"
os.makedirs(root_dir, exist_ok=True)

# =========== Baseline environment test =======================
output_dir = f"{root_dir}hie_baseline_test"
num_train_epochs = 2
cmd = f"python main_0.6.2_topk.py --bert_model bert-base-uncased --vocab_file {bert_base_vocab} --model_file {bert_base_model} " \
    f"--output_dir {output_dir} --predict_dir {output_dir} --train_file {train_file} --predict_file {dev_file} " \
    f"--max_seq_length 480 --max_query_length 50 " \
    f"--train_batch_size 8 --predict_batch_size 8 --max_answer_length 15 --num_train_epochs {num_train_epochs} " \
    f"--learning_rate {learning_rate} --max_ctx 2 --bert_name {bert_name} --task_name {task_name} --reader_name {reader_name} " \
    f"--evidence_lambda {evidence_lambda} " \
    f"--do_train --do_predict "

run_cmd(cmd)
