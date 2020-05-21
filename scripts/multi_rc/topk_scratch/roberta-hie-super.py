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

# Hierarchical super

evidence_lambda = 0.8
num_train_epochs = 8
learning_rate = 1e-6
output_dir = 'experiments/multi-rc/topk-evidence/roberta-hie-super/v1.0'

cmd = f'python main2_0.6.2_topk.py --bert_model roberta-large ' \
    f'--vocab_file {roberta_large_model_dir} --model_file {roberta_large_model_dir} ' \
    f'--output_dir {output_dir} --predict_dir {output_dir} ' \
    f'--train_file {train_file} --predict_file {dev_file} ' \
    f'--max_seq_length 512 --train_batch_size 32 --predict_batch_size 1 ' \
    f'--learning_rate {learning_rate} --num_train_epochs {num_train_epochs} ' \
    f'--fp16 --fp16_opt_level O2 --gradient_accumulation_steps 32 --per_eval_step 100 ' \
    f'--bert_name {bert_name} --task_name {task_name} --reader_name {reader_name} ' \
    f'--evidence_lambda {evidence_lambda} ' \
    f'--do_train --do_predict ' \
    f'--max_grad_norm 5.0 --adam_epsilon 1e-6 '

# run_cmd(cmd)

evidence_lambda = 0.8
num_train_epochs = 8
learning_rate = 1e-5
output_dir = 'experiments/multi-rc/topk-evidence/roberta-hie-super/v1.1'

cmd = f'python main2_0.6.2_topk.py --bert_model roberta-large ' \
    f'--vocab_file {roberta_large_model_dir} --model_file {roberta_large_model_dir} ' \
    f'--output_dir {output_dir} --predict_dir {output_dir} ' \
    f'--train_file {train_file} --predict_file {dev_file} ' \
    f'--max_seq_length 512 --train_batch_size 32 --predict_batch_size 1 ' \
    f'--learning_rate {learning_rate} --num_train_epochs {num_train_epochs} ' \
    f'--fp16 --fp16_opt_level O2 --gradient_accumulation_steps 32 --per_eval_step 100 ' \
    f'--bert_name {bert_name} --task_name {task_name} --reader_name {reader_name} ' \
    f'--evidence_lambda {evidence_lambda} ' \
    f'--do_train --do_predict ' \
    f'--max_grad_norm 5.0 --adam_epsilon 1e-6 '

# run_cmd(cmd)

evidence_lambda = 0.8
num_train_epochs = 8
learning_rate = 1e-5
output_dir = 'experiments/multi-rc/topk-evidence/roberta-hie-super/v1.2'

cmd = f'python main2_0.6.2_topk.py --bert_model roberta-large ' \
    f'--vocab_file {roberta_large_model_dir} --model_file {roberta_large_model_dir} ' \
    f'--output_dir {output_dir} --predict_dir {output_dir} ' \
    f'--train_file {train_file} --predict_file {dev_file} ' \
    f'--max_seq_length 512 --train_batch_size 32 --predict_batch_size 1 ' \
    f'--learning_rate {learning_rate} --num_train_epochs {num_train_epochs} ' \
    f'--fp16 --fp16_opt_level O2 --gradient_accumulation_steps 32 --per_eval_step 100 ' \
    f'--bert_name {bert_name} --task_name {task_name} --reader_name {reader_name} ' \
    f'--evidence_lambda {evidence_lambda} ' \
    f'--do_train --do_predict ' \
    f'--max_grad_norm 5.0 --adam_epsilon 1e-6 --patience 10 '

run_cmd(cmd)

# num_evidence = 3
# predict_cmd = f'python main_0.6.2_topk_predict_sentences.py --bert_model bert-base-uncased ' \
#     f'--vocab_file {bert_base_vocab} --model_file {bert_base_model} --output_dir {output_dir} --predict_dir {output_dir} ' \
#     f'--train_file {train_file} --predict_file {dev_file} ' \
#     f'--max_seq_length 512 --train_batch_size 32 --predict_batch_size 8 ' \
#     f'--learning_rate {learning_rate} --num_train_epochs {num_train_epochs} ' \
#     f'--fp16 --gradient_accumulation_steps 4 --per_eval_step 6000 ' \
#     f'--bert_name {bert_name} --task_name {task_name} --reader_name {reader_name} ' \
#     f'--evidence_lambda {evidence_lambda} ' \
#     f'--do_label --num_evidence {num_evidence} '
#
# # run_cmd(predict_cmd)

