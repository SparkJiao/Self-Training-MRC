import subprocess
import time
import logging
import os

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def run_cmd(cmd: str):
    logger.info(cmd)
    subprocess.check_call(cmd, shell=True)


def wait_for_file(file: str):
    if not os.path.exists(file):
        logger.info(f'Could not find file {file}. Waiting...')
        minute_cnt = 0
        while not os.path.exists(file):
            print(f'The {minute_cnt}th minute...')
            time.sleep(60)
            minute_cnt += 1
        time.sleep(60)
        logger.info(f'Find file {file} after waiting for {minute_cnt} minutes')


# model
roberta_large_model_dir = "/home/jiaofangkai/roberta-large"

train_file = '/home/jiaofangkai/multi-rc/splitv2/train.json'
dev_file = '/home/jiaofangkai/multi-rc/splitv2/dev.json'

task_name = 'topk-rc-roberta'
reader_name = 'topk-multi-rc-roberta'
bert_name = 'hie-topk-roberta'

k = 4000
label_threshold = 0.7
weight_threshold = 0.5
recurrent_times = 10
num_train_epochs = [8] * 10
sentence_id_file = None
num_evidence = 3

root_dir = f'experiments/multi-rc/topk-evidence/roberta-self-training/v3.0_acc_top{k}'
os.makedirs(root_dir, exist_ok=True)

f_handler = logging.FileHandler(os.path.join(root_dir, f'output.log'))
f_handler.setLevel(logging.INFO)
f_handler.setFormatter(logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                                         datefmt='%m/%d/%Y %H:%M:%S'))
logger.addHandler(f_handler)

logger.info('Self-training parameters:')
logger.info(f'k: {k}')
logger.info(f'label_threshold: {label_threshold}')
logger.info(f'weight_threshold: {weight_threshold}')
logger.info(f'recurrent_times: {recurrent_times}')
logger.info(f'num_evidence: {num_evidence}')

learning_rate = 1e-5
for i in range(recurrent_times):
    logger.info(f'Start running for the {i}th times.')
    output_dir = f'{root_dir}/recurrent{i}'

    if i == 0:
        evidence_lambda = 0.0
    else:
        evidence_lambda = 0.8

    cmd = f'python main2_0.6.2_topk.py --bert_model roberta-large ' \
        f'--vocab_file {roberta_large_model_dir} --model_file {roberta_large_model_dir} --output_dir {output_dir} --predict_dir {output_dir} ' \
        f'--train_file {train_file} --predict_file {dev_file} ' \
        f'--max_seq_length 512 --train_batch_size 32 --predict_batch_size 1 ' \
        f'--learning_rate {learning_rate} --num_train_epochs {num_train_epochs[i]} ' \
        f'--fp16 --fp16_opt_level O2 --gradient_accumulation_steps 32 --per_eval_step 100 ' \
        f'--bert_name {bert_name} --task_name {task_name} --reader_name {reader_name} ' \
        f'--evidence_lambda {evidence_lambda} ' \
        f'--do_label --only_correct --label_threshold {label_threshold} --weight_threshold {weight_threshold} ' \
        f'--num_evidence {num_evidence} --max_grad_norm 5.0 --adam_epsilon 1e-6 '

    cmd += ' --do_train --do_predict '

    if i == 0:
        pass
    else:
        if i > 1:
            origin_sentence_id_file = f'{root_dir}/recurrent{i - 1}/sentence_id_file.json'

            merge_cmd = f'python general_util/multi_rc/merge_multiple.py ' \
                f'--predict1 {root_dir}/recurrent0/sentence_id_file_recurrent{i - 1}.json.merge ' \
                f'--predict2 {origin_sentence_id_file} ' \
                f'--output_file {root_dir}/recurrent0/sentence_id_file_recurrent{i}.json.merge ' \
                f'--label_threshold {label_threshold} --k {k} '
            run_cmd(merge_cmd)

        sentence_id_file = f'{root_dir}/recurrent0/sentence_id_file_recurrent{i}.json.merge'
        cmd += f'--sentence_id_file {sentence_id_file}'

    run_cmd(cmd)

    if i == 0:
        run_cmd(f'python general_util/multi_rc/top_k_multiple.py --predict {root_dir}/recurrent0/sentence_id_file.json '
                f'--k {k} --label_threshold {label_threshold}')

        run_cmd(f'mv {root_dir}/recurrent0/sentence_id_file.json-top{k}-{label_threshold} '
                f'{root_dir}/recurrent0/sentence_id_file_recurrent1.json.merge')
        logger.info('=' * 50)
