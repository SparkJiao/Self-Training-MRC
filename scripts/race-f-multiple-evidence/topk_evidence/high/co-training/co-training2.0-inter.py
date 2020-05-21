import argparse
import logging
import os
import subprocess
import time

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


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

train_file = '/home/jiaofangkai/RACE/RACE/train-high.json'
dev_file = '/home/jiaofangkai/RACE/RACE/dev-high.json'
test_file = '/home/jiaofangkai/RACE/RACE/test-high.json'

task_name = 'race'
reader_name = 'multiple-race'
bert_name = 'topk-hie-race'

metric = 'accuracy'
num_train_epochs = 3

root_dir = f'experiments/race/topk-evidence/high/co-training/v2.0_acc_top{40000}'
os.makedirs(root_dir, exist_ok=True)
f_handler = logging.FileHandler(os.path.join(root_dir, f'inter_output.log'))
f_handler.setLevel(logging.INFO)
f_handler.setFormatter(logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                                         datefmt='%m/%d/%Y %H:%M:%S'))
logger.addHandler(f_handler)

sentence_id_files = [
    f'{root_dir}/recurrent6_view0-recurrent7_view1-False-inter-merge-108551-58302-166853.json',
    f'{root_dir}/recurrent6_view0-recurrent7_view1-True-inter-merge-108551-58302-108551.json'
]

num_evidence = 3
learning_rate = 4e-5
evidence_lambda = 0.8
for file_index, sentence_id_file in enumerate(sentence_id_files):
    output_dir = f'{root_dir}/label-inter-{file_index}/'
    cmd = f'python main_multi_choice_top_k_evidence.py --bert_model bert-base-uncased ' \
        f'--vocab_file {bert_base_vocab} --model_file {bert_base_model} --output_dir {output_dir} --predict_dir {output_dir} ' \
        f'--train_file {train_file} --predict_file {dev_file} --test_file {test_file} ' \
        f'--max_seq_length 380 --train_batch_size 32 --predict_batch_size 4 ' \
        f'--learning_rate {learning_rate} --num_train_epochs {num_train_epochs} ' \
        f'--fp16 --gradient_accumulation_steps 8 --per_eval_step 6000 ' \
        f'--bert_name {bert_name} --task_name {task_name} --reader_name {reader_name} ' \
        f'--evidence_lambda {evidence_lambda}  ' \
        f'--do_label --only_correct ' \
        f'--metric {metric} --num_evidence {num_evidence} ' \
        f'--sentence_id_file {sentence_id_file} '

    cmd += ' --do_train --do_predict '

    run_cmd(cmd)

