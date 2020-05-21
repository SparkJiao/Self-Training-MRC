import subprocess
import time
import logging
import os

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def run_cmd(command: str):
    logger.info(command)
    subprocess.check_call(command, shell=True)


bert_base_model = "../BERT/bert-base-uncased.tar.gz"
bert_base_vocab = "../BERT/bert-base-uncased-vocab.txt"
bert_large_model = "../BERT/bert-large-uncased.tar.gz"
bert_large_vocab = "../BERT/bert-large-uncased-vocab.txt"

train_file = '/home/jiaofangkai/RACE/RACE/train-combine.json'
dev_file = '/home/jiaofangkai/RACE/RACE/dev-combine.json'
test_file = '/home/jiaofangkai/RACE/RACE/test-combine.json'

task_name = 'race'
reader_name = 'multiple-race'
bert_name = 'hie-race-hard'

output_dir = f'experiments/race/combine/gumbel-pre-train/v1.0'

metric = 'accuracy'
learning_rate = 1e-4
num_train_epochs = 3

cmd = f'python main_multi_choice_top_k_evidence.py --bert_model bert-base-uncased ' \
      f'--vocab_file {bert_base_vocab} --model_file {bert_base_model} --output_dir {output_dir} --predict_dir {output_dir} ' \
      f'--train_file {train_file} --predict_file {dev_file} --test_file {test_file} ' \
      f'--max_seq_length 380 --train_batch_size 32 --predict_batch_size 4 ' \
      f'--learning_rate {learning_rate} --num_train_epochs {num_train_epochs} ' \
      f'--fp16 --gradient_accumulation_steps 8 --per_eval_step 6000 ' \
      f'--bert_name {bert_name} --task_name {task_name} --reader_name {reader_name} ' \
      f'--evidence_lambda 0.0  ' \
      f'--metric {metric} --do_train --do_predict --use_gumbel --freeze_bert '

# run_cmd(cmd)

bert_name = 'hie-race-reinforce'
output_dir = f'experiments/race/combine/reinforce-fine-tune/v1.0'
learning_rate = 4e-5
reward_func = 1

cmd = f'python main_multi_choice_top_k_evidence.py --bert_model bert-base-uncased ' \
      f'--vocab_file {bert_base_vocab} --model_file {bert_base_model} --output_dir {output_dir} --predict_dir {output_dir} ' \
      f'--train_file {train_file} --predict_file {dev_file} --test_file {test_file} ' \
      f'--max_seq_length 380 --train_batch_size 32 --predict_batch_size 4 ' \
      f'--learning_rate {learning_rate} --num_train_epochs {num_train_epochs} ' \
      f'--fp16 --gradient_accumulation_steps 8 --per_eval_step 6000 ' \
      f'--bert_name {bert_name} --task_name {task_name} --reader_name {reader_name} ' \
      f'--evidence_lambda 0.0  ' \
      f'--metric {metric} --do_train --do_predict --sample_steps 10 --reward_func {reward_func} ' \
      f'--pretrain experiments/race/combine/gumbel-pre-train/v1.0/pytorch_model.bin '

run_cmd(cmd)

