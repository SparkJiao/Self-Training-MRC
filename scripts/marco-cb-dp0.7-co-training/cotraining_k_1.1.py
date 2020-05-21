import logging
import os
import subprocess
import argparse
import time

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--view_id', type=int, required=True)
args = parser.parse_args()


def run_cmd(cmd: str):
    logger.info(cmd)
    subprocess.check_call(cmd, shell=True)


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

task_name = 'marco-cb-dp0.7'
reader_name = 'cb-marco'

bert_name = 'twoview'

weight_threshold = 0.8
label_threshold = 0.9
recurrent_times = 10

num_train_epochs = [2] + [3] * 20
# learning_rate = 4e-5

sentence_id_file = None

top_k = 1000
view_id = args.view_id
root_dir = f"experiments/marco-cb-dp0.7/co-training/k1.1_{top_k}"
os.makedirs(root_dir, exist_ok=True)

f_handler = logging.FileHandler(os.path.join(root_dir, f'view{view_id}-output.log'))
f_handler.setLevel(logging.INFO)
f_handler.setFormatter(logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                                         datefmt='%m/%d/%Y %H:%M:%S'))
logger.addHandler(f_handler)

logger.info('Co-training parameters:')
logger.info(f'k: {top_k}')
logger.info(f'label_threshold: {label_threshold}')
logger.info(f'weight_threshold: {weight_threshold}')
logger.info(f'recurrent_times: {recurrent_times}')

for i in range(recurrent_times):
    logger.info(f'Running at the {i}th times {view_id}th view...')

    if i == 0:
        evidence_lambda = 0.0
        learning_rate = 2e-5
    else:
        evidence_lambda = 0.8
        learning_rate = 3e-5

    output_dir = f"{root_dir}/recurrent{i}_view{view_id}/"

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
                --view_id {view_id} \
                --label_threshold {label_threshold}  "

    if i == 0:
        # need to modify if used
        pass
    elif i > 0:
        if i > 1:
            origin_sentence_id_file = f"{root_dir}/recurrent{i - 1}_view{1 - view_id}/sentence_id_file.json"
            wait_for_file(origin_sentence_id_file, time_for_writing=15)

            merge_cmd = f"python general_util/full_k/merge.py \
                                --predict1 {root_dir}/recurrent0_view{view_id}/sentence_id_file_recurrent{i - 1}.json.merge \
                                --predict2 {origin_sentence_id_file} \
                                --output_dir {root_dir}/recurrent0_view{view_id}/sentence_id_file_recurrent{i}.json.merge " \
                f"--label_threshold {label_threshold} --weight_threshold {weight_threshold} --k {top_k} "
            run_cmd(merge_cmd)
        else:
            wait_for_file(f'{root_dir}/recurrent0_view{view_id}/sentence_id_file_recurrent{i}.json.merge')

        sentence_id_file = f"{root_dir}/recurrent0_view{view_id}/sentence_id_file_recurrent{i}.json.merge"
        cmd += f"--sentence_id_files {sentence_id_file}"

    run_cmd(cmd)

    if i == 0 and view_id == 0:
        wait_for_file(f'{root_dir}/recurrent0_view1/sentence_id_file.json')

        run_cmd(f"python general_util/full_k/top_k.py --predict={root_dir}/recurrent0_view1/sentence_id_file.json --k={top_k} "
                f"--label_threshold {label_threshold} --weight_threshold {weight_threshold} ")
        run_cmd(f"python general_util/full_k/top_k.py --predict={root_dir}/recurrent0_view0/sentence_id_file.json --k={top_k} "
                f"--label_threshold {label_threshold} --weight_threshold {weight_threshold} ")
        run_cmd(
            f"mv {root_dir}/recurrent0_view1/sentence_id_file.json_top_{top_k} {root_dir}/recurrent0_view0/sentence_id_file_recurrent1.json.merge")
        run_cmd(
            f"mv {root_dir}/recurrent0_view0/sentence_id_file.json_top_{top_k} {root_dir}/recurrent0_view1/sentence_id_file_recurrent1.json.merge")
    logger.info('=' * 50)
