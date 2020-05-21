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


# model
bert_base_model = "../BERT/bert-base-uncased.tar.gz"
bert_base_vocab = "../BERT/bert-base-uncased-vocab.txt"
bert_large_model = "../BERT/bert-large-uncased.tar.gz"
bert_large_vocab = "../BERT/bert-large-uncased-vocab.txt"

train_file = '/home/jiaofangkai/dream/data/train_in_race.json'
dev_file = '/home/jiaofangkai/dream/data/dev_in_race.json'
test_file = '/home/jiaofangkai/dream/data/test_in_race.json'

task_name = 'dream'
reader_name = 'multiple-race'
bert_name = 'multiple-hie-race'

metric = 'accuracy'
evidence_lambda = 0.0

learning_rate = 3e-5
num_train_epochs = 3
output_dir = 'experiments/dream/multiple-evidence/scratch/hie1.0'

cmd = f'python main_multi_choice_multiple_evidence.py --bert_model bert-base-uncased ' \
    f'--vocab_file {bert_base_vocab} --model_file {bert_base_model} --output_dir {output_dir} --predict_dir {output_dir} ' \
    f'--train_file {train_file} --predict_file {dev_file} --test_file {test_file} ' \
    f'--max_seq_length 400 --train_batch_size 32 --predict_batch_size 4 ' \
    f'--learning_rate {learning_rate} --num_train_epochs {num_train_epochs} ' \
    f'--fp16 --gradient_accumulation_steps 8 --per_eval_step 3000 ' \
    f'--bert_name {bert_name} --task_name {task_name} --reader_name {reader_name} ' \
    f'--evidence_lambda {evidence_lambda} --num_choices 3 ' \
    f'--metric {metric} ' \
    f'--do_train --do_predict'

# run_cmd(cmd)

# ===========================================================
# Best performance
learning_rate = 2e-5
num_train_epochs = 3
output_dir = 'experiments/dream/multiple-evidence/scratch/hie1.1'

cmd = f'python main_multi_choice_multiple_evidence.py --bert_model bert-base-uncased ' \
    f'--vocab_file {bert_base_vocab} --model_file {bert_base_model} --output_dir {output_dir} --predict_dir {output_dir} ' \
    f'--train_file {train_file} --predict_file {dev_file} --test_file {test_file} ' \
    f'--max_seq_length 400 --train_batch_size 32 --predict_batch_size 4 ' \
    f'--learning_rate {learning_rate} --num_train_epochs {num_train_epochs} ' \
    f'--fp16 --gradient_accumulation_steps 8 --per_eval_step 3000 ' \
    f'--bert_name {bert_name} --task_name {task_name} --reader_name {reader_name} ' \
    f'--evidence_lambda {evidence_lambda} --num_choices 3 ' \
    f'--metric {metric} ' \
    f'--do_train --do_predict'

# run_cmd(cmd)

# ===========================================================

learning_rate = 4e-5
num_train_epochs = 2
output_dir = 'experiments/dream/multiple-evidence/scratch/hie1.2'

cmd = f'python main_multi_choice_multiple_evidence.py --bert_model bert-base-uncased ' \
    f'--vocab_file {bert_base_vocab} --model_file {bert_base_model} --output_dir {output_dir} --predict_dir {output_dir} ' \
    f'--train_file {train_file} --predict_file {dev_file} --test_file {test_file} ' \
    f'--max_seq_length 400 --train_batch_size 32 --predict_batch_size 4 ' \
    f'--learning_rate {learning_rate} --num_train_epochs {num_train_epochs} ' \
    f'--fp16 --gradient_accumulation_steps 8 --per_eval_step 3000 ' \
    f'--bert_name {bert_name} --task_name {task_name} --reader_name {reader_name} ' \
    f'--evidence_lambda {evidence_lambda} --num_choices 3 ' \
    f'--metric {metric} ' \
    f'--do_train --do_predict'

# run_cmd(cmd)

# ===========================================================

learning_rate = 3e-5
num_train_epochs = 2
output_dir = 'experiments/dream/multiple-evidence/scratch/hie1.3'

cmd = f'python main_multi_choice_multiple_evidence.py --bert_model bert-base-uncased ' \
    f'--vocab_file {bert_base_vocab} --model_file {bert_base_model} --output_dir {output_dir} --predict_dir {output_dir} ' \
    f'--train_file {train_file} --predict_file {dev_file} --test_file {test_file} ' \
    f'--max_seq_length 400 --train_batch_size 32 --predict_batch_size 4 ' \
    f'--learning_rate {learning_rate} --num_train_epochs {num_train_epochs} ' \
    f'--fp16 --gradient_accumulation_steps 8 --per_eval_step 3000 ' \
    f'--bert_name {bert_name} --task_name {task_name} --reader_name {reader_name} ' \
    f'--evidence_lambda {evidence_lambda} --num_choices 3 ' \
    f'--metric {metric} ' \
    f'--do_train --do_predict'

# run_cmd(cmd)

# ===========================================================

learning_rate = 1e-5
num_train_epochs = 3
output_dir = 'experiments/dream/multiple-evidence/scratch/hie1.4'

cmd = f'python main_multi_choice_multiple_evidence.py --bert_model bert-base-uncased ' \
    f'--vocab_file {bert_base_vocab} --model_file {bert_base_model} --output_dir {output_dir} --predict_dir {output_dir} ' \
    f'--train_file {train_file} --predict_file {dev_file} --test_file {test_file} ' \
    f'--max_seq_length 400 --train_batch_size 32 --predict_batch_size 4 ' \
    f'--learning_rate {learning_rate} --num_train_epochs {num_train_epochs} ' \
    f'--fp16 --gradient_accumulation_steps 8 --per_eval_step 3000 ' \
    f'--bert_name {bert_name} --task_name {task_name} --reader_name {reader_name} ' \
    f'--evidence_lambda {evidence_lambda} --num_choices 3 ' \
    f'--metric {metric} ' \
    f'--do_train --do_predict'

run_cmd(cmd)
