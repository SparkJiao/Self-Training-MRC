import logging
import os
import subprocess
import time

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


# parser = argparse.ArgumentParser()
# parser.add_argument('--view_id', type=int, required=True)
# args = parser.parse_args()


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

train_file = '../BERT/max_f1/coqa-train-v1.0.json'
# dev_file = '../BERT/max_f1/coqa-dev-v1.0.json'
dev_file = '../BERT/max_f1/coqa-dev-single-sent.json'

task_name = 'coqa-top-k'
reader_name = 'coqa-top-k'
bert_name = 'hie-topk-32'

weight_threshold = 0.6
label_threshold = 0.9
recurrent_times = 10
num_train_epochs = [3] * 10
num_evidence = 1

sentence_id_file = None

top_k = 1500
root_dir = f"experiments/coqa/topk-self-training/1.0_{top_k}shaod"
os.makedirs(root_dir, exist_ok=True)

f_handler = logging.FileHandler(os.path.join(root_dir, f'output.log'))
f_handler.setLevel(logging.INFO)
f_handler.setFormatter(logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                                         datefmt='%m/%d/%Y %H:%M:%S'))
logger.addHandler(f_handler)

logger.info('Self-training parameters:')
logger.info(f'k: {top_k}')
logger.info(f'label_threshold: {label_threshold}')
logger.info(f'weight_threshold: {weight_threshold}')
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
    predict_dir = f"{output_dir}/single_sent"

    cmd = f"python main_0.6.2_topk.py --bert_model bert-base-uncased --vocab_file {bert_base_vocab} --model_file {bert_base_model} " \
        f"--output_dir {output_dir} --predict_dir {predict_dir} --train_file {train_file} --predict_file {dev_file} " \
        f"--max_seq_length 512 --max_query_length 385 " \
        f"--train_batch_size 6 --predict_batch_size 6 --max_answer_length 15 --num_train_epochs {num_train_epochs[i]} " \
        f"--learning_rate {learning_rate} --max_ctx 2 --bert_name {bert_name} --task_name {task_name} --reader_name {reader_name} " \
        f"--evidence_lambda {evidence_lambda} " \
        f"--num_evidence {num_evidence} "

    cmd += '--do_predict '

    run_cmd(cmd)

    logger.info('=' * 50)
