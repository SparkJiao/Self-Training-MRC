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
bert_base_model = "../BERT/bert-base-uncased.tar.gz"
bert_base_vocab = "../BERT/bert-base-uncased-vocab.txt"
bert_large_model = "../BERT/bert-large-uncased.tar.gz"
bert_large_vocab = "../BERT/bert-large-uncased-vocab.txt"

train_file = '/home/jiaofangkai/multi-rc/splitv2/train.json'
dev_file = '/home/jiaofangkai/multi-rc/splitv2/dev.json'

task_name = 'topk-rc'
reader_name = 'multi-rc-topk-search'
bert_name = 'hie-topk'

k = 2000
label_threshold = 0.8
weight_threshold = 0.5
recurrent_times = 10
num_train_epochs = [8] * 10
sentence_id_file = None
num_evidence = 3

root_dir = f'experiments/multi-rc/topk-evidence/self-training/v2.0_acc_top{k}'
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

learning_rate = 2e-5
for i in range(recurrent_times):
    logger.info(f'Start running for the {i}th times.')
    output_dir = f'{root_dir}/recurrent{i}'
    predict_dir = f'{output_dir}/sent3_predict'
    evidence_search_file = f'{output_dir}/sent2_predict/evidence_id.json'

    if i == 0:
        evidence_lambda = 0.0
    else:
        evidence_lambda = 0.8

    cmd = f'python main_0.6.2_topk.py --bert_model bert-base-uncased ' \
          f'--vocab_file {bert_base_vocab} --model_file {bert_base_model} --output_dir {output_dir} --predict_dir {predict_dir} ' \
          f'--train_file {train_file} --predict_file {dev_file} ' \
          f'--max_seq_length 512 --train_batch_size 32 --predict_batch_size 8 ' \
          f'--learning_rate {learning_rate} --num_train_epochs {num_train_epochs[i]} ' \
          f'--fp16 --gradient_accumulation_steps 4 --per_eval_step 6000 ' \
          f'--bert_name {bert_name} --task_name {task_name} --reader_name {reader_name} ' \
          f'--evidence_lambda {evidence_lambda} ' \
          f'--num_evidence {num_evidence} ' \
          f'--evidence_search_file {evidence_search_file}'

    cmd += ' --do_predict '

    run_cmd(cmd)
