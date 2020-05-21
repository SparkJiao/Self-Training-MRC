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
bert_name = 'topk-hie-race'

metric = 'accuracy'
evidence_lambda = 0.8

num_train_epochs = 3
learning_rate = [2e-5, 3e-5]

sentence_dir = 'experiments/dream/topk-evidence/co-training/v1.1_acc_top3000/'
sentence_id_files = [
    os.path.join(sentence_dir, 'recurrent11_view0-recurrent11_view1-False-inter-merge-8587-7096-15645.json'),
    os.path.join(sentence_dir, 'recurrent11_view0-recurrent11_view1-True-inter-merge-8587-7096-8587.json'),
    os.path.join(sentence_dir, 'recurrent8_view0-recurrent11_view1-False-inter-merge-8372-6964-15320.json'),
    os.path.join(sentence_dir, 'recurrent8_view0-recurrent11_view1-True-inter-merge-8372-6964-8372.json')
]

for lr in learning_rate:
    for file_index, sentence_id_file in enumerate(sentence_id_files):

        output_dir = f'experiments/dream/topk-evidence/co-training/v1.1-inter-{lr}-{file_index}/'

        cmd = f'python main_multi_choice_top_k_evidence.py --bert_model bert-base-uncased ' \
            f'--vocab_file {bert_base_vocab} --model_file {bert_base_model} --output_dir {output_dir} --predict_dir {output_dir} ' \
            f'--train_file {train_file} --predict_file {dev_file} --test_file {test_file} ' \
            f'--max_seq_length 400 --train_batch_size 32 --predict_batch_size 4 ' \
            f'--learning_rate {lr} --num_train_epochs {num_train_epochs} ' \
            f'--fp16 --gradient_accumulation_steps 8 --per_eval_step 3000 ' \
            f'--bert_name {bert_name} --task_name {task_name} --reader_name {reader_name} ' \
            f'--evidence_lambda {evidence_lambda} --num_choices 3 ' \
            f'--metric {metric} ' \
            f'--do_train --do_predict --do_label ' \
            f'--sentence_id_file {sentence_id_file}'

        run_cmd(cmd)
