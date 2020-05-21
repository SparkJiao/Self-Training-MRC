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
reader_name = 'topk-multi-rc'
bert_name = 'hie-topk'

evidence_lambda = 0.8
num_train_epochs = 3
learning_rate = 2e-5
root_dir = 'experiments/multi-rc/topk-evidence/rule/'

os.makedirs(root_dir, exist_ok=True)
gen_cmd = f'python scripts/gen_evidence_by_rule.py --task_name MultiRC --input_file {train_file} --top_k {5000} ' \
    f'--num_evidences {3} --output_file {root_dir}sentence_id_rule1.0-5000-3.json'
run_cmd(gen_cmd)

output_dir = f'{root_dir}v1.0/'
rule_sentence_id_file = f'{root_dir}sentence_id_rule1.0-5000-3.json'

cmd = f'python main_0.6.2_topk.py --bert_model bert-base-uncased ' \
    f'--vocab_file {bert_base_vocab} --model_file {bert_base_model} --output_dir {output_dir} --predict_dir {output_dir} ' \
    f'--train_file {train_file} --predict_file {dev_file} ' \
    f'--max_seq_length 512 --train_batch_size 32 --predict_batch_size 8 ' \
    f'--learning_rate {learning_rate} --num_train_epochs {num_train_epochs} ' \
    f'--fp16 --gradient_accumulation_steps 4 --per_eval_step 6000 ' \
    f'--bert_name {bert_name} --task_name {task_name} --reader_name {reader_name} ' \
    f'--evidence_lambda {evidence_lambda} ' \
    f'--do_train --do_predict ' \
    f'--do_label --sentence_id_file {rule_sentence_id_file} '

run_cmd(cmd)

