import logging
import os

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

bert_base_model = "../BERT/bert-base-uncased.tar.gz"
bert_base_vocab = "../BERT/bert-base-uncased-vocab.txt"
bert_large_model = "../BERT/bert-large-uncased.tar.gz"
bert_large_vocab = "../BERT/bert-large-uncased-vocab.txt"
train_file = "data-set/coqa-train-v1.0.json"
dev_file = "data-set/coqa-dev-v1.0.json"

reader_name = 'coqa-sen'
do_negative_sampling = True
read_extra_self = True
multi_inputs = True
sample_ratio = 0.5
task_name = 'coqa-sen-{}-{}-{}'.format(do_negative_sampling, read_extra_self, sample_ratio)


output_dir = f"experiments/negative_sample/coqa-coqa-1.8/"

cmd = f"python transfer_main.py --bert_model bert-base-uncased --vocab_file {bert_base_vocab} --model_file {bert_base_model} " \
    f"--output_dir {output_dir} --predict_dir {output_dir} --train_file {train_file} --predict_file {dev_file} " \
    f"--max_seq_length 512 --max_query_length 385 --do_train --do_predict --train_batch_size 6 --predict_batch 6 " \
    f"--learning_rate 3e-5 --num_train_epochs 3.0 --max_answer_length 15 --max_ctx 2 " \
    f"--task_name {task_name} --bert_name hie-neg --reader_name {reader_name} --evidence_lambda 0.0 " \
    f"--sample_ratio {sample_ratio} --negative_lambda 0.8 "

if do_negative_sampling:
    cmd += '--do_negative_sampling '
if read_extra_self:
    cmd += "--read_extra_self "
if multi_inputs:
    cmd += '--multi_inputs'

logger.debug(cmd)
os.system(cmd)
