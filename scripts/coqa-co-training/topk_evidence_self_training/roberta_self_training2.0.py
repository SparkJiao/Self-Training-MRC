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
roberta_large_model_dir = "/home/jiaofangkai/roberta-large"

train_file = '../BERT/max_f1/coqa-train-v1.0.json'
dev_file = '../BERT/max_f1/coqa-dev-v1.0.json'

task_name = 'coqa-top-k-roberta'
reader_name = 'coqa-top-k-roberta'
bert_name = 'hie-topk-roberta'

weight_threshold = 0.4
label_threshold = 0.7
recurrent_times = 10
num_train_epochs = [3] * 10
num_evidence = 1

sentence_id_file = None

top_k = 3000
root_dir = f"experiments/coqa/topk-self-training/roberta_2.0_{top_k}"
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
        learning_rate = 1e-5
    else:
        learning_rate = 1e-5
        evidence_lambda = 0.8

    output_dir = f"{root_dir}/recurrent{i}/"

    cmd = f"python main2_0.6.2_topk.py --bert_model roberta-large " \
          f" --vocab_file {roberta_large_model_dir} --model_file {roberta_large_model_dir} " \
          f"--output_dir {output_dir} --predict_dir {output_dir} --train_file {train_file} --predict_file {dev_file} " \
          f"--max_seq_length 512 --max_query_length 385 " \
          f"--train_batch_size 32 --predict_batch_size 1 --max_answer_length 15 " \
          f"--num_train_epochs {num_train_epochs[i]} --gradient_accumulation_steps 32 " \
          f"--learning_rate {learning_rate} --max_ctx 2 --bert_name {bert_name} --task_name {task_name} --reader_name {reader_name} " \
          f"--evidence_lambda {evidence_lambda} " \
          f"--do_label --weight_threshold {weight_threshold} --only_correct --label_threshold {label_threshold} " \
          f"--num_evidence {num_evidence} " \
          f"--max_grad_norm 5.0 --fp16 --fp16_opt_level O2 --per_eval_step 100 --adam_epsilon 1e-6 "

    # if i > 0:
    cmd += '--do_train --do_predict '

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
        cmd += f"--sentence_id_file {sentence_id_file}"

    run_cmd(cmd)

    if i == 0:
        run_cmd(
            f"python general_util/full_k/top_k.py --predict={root_dir}/recurrent0/sentence_id_file.json --k={top_k} "
            f"--label_threshold {label_threshold} --weight_threshold {weight_threshold} ")

        run_cmd(
            f"mv {root_dir}/recurrent0/sentence_id_file.json_top_{top_k} "
            f"{root_dir}/recurrent0/sentence_id_file_recurrent1.json.merge")
    logger.info('=' * 50)
