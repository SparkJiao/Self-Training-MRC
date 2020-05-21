import argparse
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


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, required=True)
args = parser.parse_args()


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
bert_name = 'hie-hard-16'

evidence_lambda = 0.0
num_train_epochs = 6
learning_rate = 1e-4
output_dir = f'experiments/multi-rc/gumbel-pre-train/v1.0_seed{args.seed}'

cmd = f'python main_0.6.2_topk.py --bert_model bert-base-uncased ' \
      f'--vocab_file {bert_base_vocab} --model_file {bert_base_model} --output_dir {output_dir} --predict_dir {output_dir} ' \
      f'--train_file {train_file} --predict_file {dev_file} ' \
      f'--max_seq_length 512 --train_batch_size 32 --predict_batch_size 8 ' \
      f'--learning_rate {learning_rate} --num_train_epochs {num_train_epochs} ' \
      f'--fp16 --gradient_accumulation_steps 4 --per_eval_step 6000 ' \
      f'--bert_name {bert_name} --task_name {task_name} --reader_name {reader_name} ' \
      f'--evidence_lambda {evidence_lambda} ' \
      f'--do_train --do_predict --use_gumbel --freeze_bert --seed {args.seed} '

run_cmd(cmd)

task_name = 'topk-rc'
reader_name = 'topk-multi-rc'
bert_name = 'hie-reinforce-16'

evidence_lambda = 0.0
num_train_epochs = 10
learning_rate = 2e-5
reward_func = 1
output_dir = f'experiments/multi-rc/reinforce-fine-tune/v1.0_seed{args.seed}'

cmd = f'python main_0.6.2_topk.py --bert_model bert-base-uncased ' \
      f'--vocab_file {bert_base_vocab} --model_file {bert_base_model} --output_dir {output_dir} --predict_dir {output_dir} ' \
      f'--train_file {train_file} --predict_file {dev_file} ' \
      f'--max_seq_length 512 --train_batch_size 32 --predict_batch_size 8 ' \
      f'--learning_rate {learning_rate} --num_train_epochs {num_train_epochs} ' \
      f'--fp16 --gradient_accumulation_steps 4 --per_eval_step 6000 ' \
      f'--bert_name {bert_name} --task_name {task_name} --reader_name {reader_name} ' \
      f'--evidence_lambda {evidence_lambda} ' \
      f'--do_train --do_predict --sample_steps 10 --reward_func {reward_func} ' \
      f'--pretrain experiments/multi-rc/gumbel-pre-train/v1.0_seed{args.seed}/pytorch_model.bin --seed {args.seed} '

run_cmd(cmd)
