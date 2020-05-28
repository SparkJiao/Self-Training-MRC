import logging
import os
import subprocess
import time

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def run_cmd(command: str):
    logger.info(command)
    subprocess.check_call(command, shell=True)


def wait_for_file(file: str, minute: int = 1):
    if not os.path.exists(file):
        logger.info(f'Could not find file {file}. Waiting...')
        minute_cnt = 0
        while not os.path.exists(file):
            print(f'The {minute_cnt}th minute...')
            time.sleep(60)
            minute_cnt += 1
        print(f'Have found file. Wait for writing for {minute} extra minutes...')
        time.sleep(60 * minute)
        logger.info(f'Find file {file} after waiting for {minute_cnt + minute} minutes')


# model
bert_base_model = "~/bert-base-uncased.tar.gz"
bert_base_vocab = "~/bert-base-uncased-vocab.txt"
# bert_large_model = "../BERT/bert-large-uncased.tar.gz"
# bert_large_vocab = "../BERT/bert-large-uncased-vocab.txt"

train_file = 'data/RACE/train-middle-ini.json'
dev_file = 'data/RACE/dev-middle.json'
test_file = 'data/RACE/test-middle.json'

task_name = 'race'
reader_name = 'multiple-race'
bert_name = 'topk-hie-race'

learning_rate = 4e-5
num_train_epochs = 3

metric = 'accuracy'

os.makedirs('experiments/race/topk-evidence/middle/rule/', exist_ok=True)
gen_cmd = f'python scripts/gen_evidence_by_rule.py --task_name RACE --input_file {train_file} --top_k {70000} ' \
    f'--num_evidences {2} --output_file experiments/race/topk-evidence/middle/rule/sentence_id_rule1.0-70000-2.json'
run_cmd(gen_cmd)

output_dir = f'experiments/race/topk-evidence/middle/rule/idf_v1.0'
rule_sentence_id_file = 'experiments/race/topk-evidence/middle/rule/sentence_id_rule1.0-70000-2.json'

cmd = f'python main_multi_choice_top_k_evidence.py --bert_model bert-base-uncased ' \
    f'--vocab_file {bert_base_vocab} --model_file {bert_base_model} --output_dir {output_dir} --predict_dir {output_dir} ' \
    f'--train_file {train_file} --predict_file {dev_file} --test_file {test_file} ' \
    f'--max_seq_length 380 --train_batch_size 32 --predict_batch_size 4 ' \
    f'--learning_rate {learning_rate} --num_train_epochs {num_train_epochs} ' \
    f'--fp16 --gradient_accumulation_steps 8 --per_eval_step 3000 ' \
    f'--bert_name {bert_name} --task_name {task_name} --reader_name {reader_name} ' \
    f'--metric {metric} ' \
    f'--do_label --sentence_id_file {rule_sentence_id_file} --evidence_lambda 0.8 '

cmd += ' --do_train --do_predict '

run_cmd(cmd)
