import logging
import os
import subprocess
import argparse
import time

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def run_cmd(cmd: str):
    logger.info(cmd)
    subprocess.check_call(cmd, shell=True)


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

weight_threshold = 0.8
label_threshold = 0.9
recurrent_times = 10

num_train_epochs = [2] + [3] * 20
# learning_rate = 4e-5

sentence_id_file = None

top_k = 1000
root_dir = f"experiments/marco-cb-dp0.7/self-training/k1.1_{top_k}"
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
    logger.info(f'Running at the {i}th times ...')

    if i == 0:
        evidence_lambda = 0.0
        learning_rate = 2e-5
    else:
        evidence_lambda = 0.8
        learning_rate = 3e-5

    output_dir = f"{root_dir}/recurrent{i}/"

    cmd = f"python main_0.6.2.py \
                --bert_model bert-base-uncased \
                --vocab_file {bert_base_vocab} \
                --model_file {bert_base_model} \
                --output_dir {output_dir} \
                --predict_dir {output_dir} \
                --train_file {train_file} \
                --predict_file {dev_file} \
                --max_seq_length 480 --max_query_length 50 \
                --do_train --do_predict --train_batch_size 8 --predict_batch_size 8 --max_answer_length 15 \
                --num_train_epochs {num_train_epochs[i]} \
                --learning_rate {learning_rate} \
                --max_ctx 3 \
                --bert_name {bert_name} \
                --task_name {task_name} \
                --reader_name {reader_name} \
                --evidence_lambda {evidence_lambda} \
                --do_label \
                --weight_threshold {weight_threshold} \
                --only_correct \
                --label_threshold {label_threshold}  "

    if i == 0:
        # need to modify if used
        pass
    elif i > 0:
        if i > 1:
            origin_sentence_id_file = f"{root_dir}/recurrent{i - 1}/sentence_id_file.json"

            merge_cmd = f"python general_util/full_k/merge.py " \
                f"--predict1 {root_dir}/recurrent0/sentence_id_file_recurrent{i - 1}.json.merge " \
                f"--predict2 {origin_sentence_id_file} " \
                f"--output_dir {root_dir}/recurrent0/sentence_id_file_recurrent{i}.json.merge " \
                f"--label_threshold {label_threshold} --weight_threshold {weight_threshold} --k {top_k} "
            run_cmd(merge_cmd)

        sentence_id_file = f"{root_dir}/recurrent0/sentence_id_file_recurrent{i}.json.merge"
        cmd += f"--sentence_id_files {sentence_id_file}"

    run_cmd(cmd)

    if i == 0:

        run_cmd(f"python general_util/full_k/top_k.py --predict={root_dir}/recurrent0/sentence_id_file.json --k={top_k} "
                f"--label_threshold {label_threshold} --weight_threshold {weight_threshold} ")
        run_cmd(
            f"mv {root_dir}/recurrent0/sentence_id_file.json_top_{top_k} "
            f"{root_dir}/recurrent0/sentence_id_file_recurrent1.json.merge")
    logger.info('=' * 50)
