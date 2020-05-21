import argparse
import logging
import os
import subprocess
import time

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--view_id', type=int, required=True)
args = parser.parse_args()
view_id = args.view_id


def run_cmd(cmd: str):
    logger.info(cmd)
    subprocess.check_call(cmd, shell=True)


def wait_for_file(file: str, minute: int = 1):
    if not os.path.exists(file):
        logger.info(f'Could not find file {file}. Waiting...')
        minute_cnt = 0
        while not os.path.exists(file):
            print(f'The {minute_cnt}th minute...')
            time.sleep(60)
            minute_cnt += 1
        print(f'Have found file. Wait for writing for {minute} extra minutes...')
        time.sleep(60 * minute)
        logger.info(f'Find file {file} after waiting for {minute_cnt + minute} minutes')


# model
bert_base_model = "../BERT/bert-base-uncased.tar.gz"
bert_base_vocab = "../BERT/bert-base-uncased-vocab.txt"
bert_large_model = "../BERT/bert-large-uncased.tar.gz"
bert_large_vocab = "../BERT/bert-large-uncased-vocab.txt"

train_file = '/home/jiaofangkai/RACE/RACE/train-combine.json'
dev_file = '/home/jiaofangkai/RACE/RACE/dev-combine.json'
test_file = '/home/jiaofangkai/RACE/RACE/test-combine.json'

task_name = 'race'
reader_name = 'multiple-race'
bert_name = 'topk-two-view-race'

metric = 'accuracy'

k = 50000
label_threshold = 0.9
weight_threshold = 0.5

num_train_epochs = [3] * 10
sentence_id_file = None

root_dir = f'experiments/race/topk-evidence/combine/co-training/v2.0_acc_top{k}'
os.makedirs(root_dir, exist_ok=True)

f_handler = logging.FileHandler(os.path.join(root_dir, f'view{view_id}_predict_output.log'))
f_handler.setLevel(logging.INFO)
f_handler.setFormatter(logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                                         datefmt='%m/%d/%Y %H:%M:%S'))
logger.addHandler(f_handler)

num_evidence = 3
predict_recurrent_list = [0, 7, 8]
for i in predict_recurrent_list:
    logger.info(f'Start predicting for the {i}th times and the {view_id}th view')
    output_dir = f'{root_dir}/recurrent{i}_view{view_id}'
    predict_dir = f'{root_dir}/recurrent{i}_view{view_id}/dev_predict/'

    if i == 0:
        evidence_lambda = 0.0
        learning_rate = 4e-5
    else:
        evidence_lambda = 0.8
        learning_rate = 4e-5

    cmd = f'python main_multi_choice_top_k_evidence.py --bert_model bert-base-uncased ' \
        f'--vocab_file {bert_base_vocab} --model_file {bert_base_model} --output_dir {output_dir} --predict_dir {predict_dir} ' \
        f'--train_file {train_file} --predict_file {dev_file} --test_file {dev_file} ' \
        f'--max_seq_length 380 --train_batch_size 32 --predict_batch_size 4 ' \
        f'--learning_rate {learning_rate} --num_train_epochs {num_train_epochs[i]} ' \
        f'--fp16 --gradient_accumulation_steps 8 --per_eval_step 6000 ' \
        f'--bert_name {bert_name} --task_name {task_name} --reader_name {reader_name} ' \
        f'--evidence_lambda {evidence_lambda}  ' \
        f'--metric {metric} --num_evidence {num_evidence} --view_id {view_id} '

    cmd += ' --do_predict '

    run_cmd(cmd)

