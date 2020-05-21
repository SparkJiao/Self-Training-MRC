import logging
import os
import argparse

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
# model
bert_base_model = "./bert-base-uncased.tar.gz"
bert_base_vocab = "./bert-base-uncased-vocab.txt"
bert_large_model = "./bert-large-uncased.tar.gz"
bert_large_vocab = "./bert-large-uncased-vocab.txt"
train_file = './boolq-enlarge/train.jsonl.enlarge'
dev_file = './boolq-enlarge/dev.jsonl.enlarge'

task_name = 'boolq'
reader_name = 'boolq'

bert_name = 'hie-topk-32'

weight_threshold = 0.5
label_threshold = 0.7
recurrent_times = 12

num_train_epochs = [4] * 12
learning_rate = [3e-5] * 12

sentence_id_file = None

num_evidence = 1
power_length = 1

top_k = 500
root_dir = f"experiments/boolq-topk-multiEvidence-selftraining/v4.0_{top_k}_enlarge_{num_evidence}evidence"

def run_cmd(cmd):
    os.system(cmd)

for i in range(recurrent_times):
    logger.info(f'Running at the {i}th times...')

    if i == 0:
        evidence_lambda = 0.0
    else:
        evidence_lambda = 0.8

    output_dir = f"{root_dir}/recurrent{i}/"

    cmd = f"python main_0.6.2_topk.py --bert_model bert-base-uncased --vocab_file {bert_base_vocab} --model_file {bert_base_model} " \
        f"--output_dir {output_dir} --predict_dir {output_dir} --train_file {train_file} --predict_file {dev_file} " \
        f"--max_seq_length 512 --max_query_length 80 " \
        f"--train_batch_size 6 --predict_batch_size 6 --max_answer_length 15 --num_train_epochs {num_train_epochs[i]} " \
        f"--learning_rate {learning_rate[i]} --max_ctx 2 --bert_name {bert_name} --task_name {task_name} --reader_name {reader_name} " \
        f"--evidence_lambda {evidence_lambda} " \
        f"--do_label --weight_threshold {weight_threshold} --only_correct --label_threshold {label_threshold} " \
        f"--num_evidence {num_evidence} "

    cmd += '--do_train --do_predict '

    if i == 0:
        # need to modify if used
        pass
    elif i > 0:
        if i > 1:
            origin_sentence_id_file = f"{root_dir}/recurrent{i - 1}/sentence_id_file.json"

            merge_cmd = f"python general_util/full_k/merge.py " \
                f"--predict1 {root_dir}/recurrent0/sentence_id_file_recurrent{i - 1}.json.merge " \
                f"--predict2 {origin_sentence_id_file} " \
                f"--output_dir {root_dir}/recurrent0/sentence_id_file_recurrent{i}.json.merge " \
                f"--label_threshold {label_threshold} --weight_threshold {weight_threshold} --k {top_k} "
            run_cmd(merge_cmd)

        sentence_id_file = f"{root_dir}/recurrent0/sentence_id_file_recurrent{i}.json.merge"
        cmd += f"--sentence_id_file {sentence_id_file}"

    run_cmd(cmd)

    if i == 0:
        run_cmd(f"python general_util/full_k/top_k.py --predict={root_dir}/recurrent0/sentence_id_file.json --k={top_k} "
                f"--label_threshold {label_threshold} --weight_threshold {weight_threshold} ")

        run_cmd(
            f"mv {root_dir}/recurrent0/sentence_id_file.json_top_{top_k} "
            f"{root_dir}/recurrent0/sentence_id_file_recurrent1.json.merge")
    logger.info('=' * 50)

