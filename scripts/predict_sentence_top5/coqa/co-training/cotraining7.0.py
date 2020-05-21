import logging
import os
import subprocess

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
# model
bert_base_model = "../BERT/bert-base-uncased.tar.gz"
bert_base_vocab = "../BERT/bert-base-uncased-vocab.txt"
bert_large_model = "../BERT/bert-large-uncased.tar.gz"
bert_large_vocab = "../BERT/bert-large-uncased-vocab.txt"

train_file = '../BERT/max_f1/coqa-train-v1.0.json'
dev_file = '../BERT/max_f1/coqa-dev-v1.0.json'

task_name = 'coqa'
reader_name = 'coqa'

bert_name = 'twoview'

weight_threshold = 0.6
label_threshold = 0.5
recurrent_times = 10

num_train_epochs = [3] * 21
# learning_rate = 2e-5

sentence_id_file = None

view_num = 2

top_k = 1000
root_dir = f"experiments/coqa/co-training/v7.0_{top_k}"


def run_cmd(cmd: str):
    logger.info(cmd)
    subprocess.check_call(cmd, shell=True)


for i in range(recurrent_times):
    for view_id in range(view_num):
        logger.info(f'Running at the {i}th times {view_id}th view...')

        if i == 0:
            evidence_lambda = 0.0
            learning_rate = 2e-5
        else:
            evidence_lambda = 0.8
            learning_rate = 3e-5

        output_dir = f"{root_dir}/recurrent{i}_view{view_id}/"

        cmd = f"python predict_sentence_main0.6.2.py \
                --bert_model bert-base-uncased \
                --vocab_file {bert_base_vocab} \
                --model_file {bert_base_model} \
                --output_dir {output_dir} \
                --predict_dir {output_dir} \
                --train_file {train_file} \
                --predict_file {dev_file} \
                --max_seq_length 512 --max_query_length 385 \
                --do_predict --train_batch_size 6 --predict_batch_size 6 --max_answer_length 15 \
                --num_train_epochs {num_train_epochs[i]} \
                --learning_rate {learning_rate} \
                --max_ctx 2 \
                --bert_name {bert_name} \
                --task_name {task_name} \
                --reader_name {reader_name} \
                --evidence_lambda {evidence_lambda} \
                --do_label \
                --weight_threshold {weight_threshold} \
                --only_correct \
                --view_id {view_id} \
                --label_threshold {label_threshold}  "

        # if i == 0:
        #     # need to modify if used
        #     pass
        # elif i > 0:
        #     if i > 1:
        #         origin_sentence_id_file = f"{root_dir}/recurrent{i - 1}_view{1 - view_id}/sentence_id_file.json"
        #         merge_cmd = f"python merge.py \
        #                         --predict1 {root_dir}/recurrent0_view{view_id}/sentence_id_file_recurrent{i - 1}.json.merge \
        #                         --predict2 {origin_sentence_id_file} \
        #                         --output_dir {root_dir}/recurrent0_view{view_id}/sentence_id_file_recurrent{i}.json.merge"
        #         os.system(merge_cmd)
        #
        #     sentence_id_file = f"{root_dir}/recurrent0_view{view_id}/sentence_id_file_recurrent{i}.json.merge"
        #     cmd += f"--sentence_id_files {sentence_id_file}"

        run_cmd(cmd)

        # if not os.path.exists(os.path.join(output_dir, 'pytorch_model.bin')):
        #     raise RuntimeError('Last command didn\'t finish.')
        #
        # if i == 0 and view_id == 1:
        #     run_cmd(f"python top_k.py --predict={root_dir}/recurrent0_view1/sentence_id_file.json --k={top_k}")
        #     run_cmd(f"python top_k.py --predict={root_dir}/recurrent0_view0/sentence_id_file.json --k={top_k}")
        #     run_cmd(
        #         f"mv {root_dir}/recurrent0_view1/sentence_id_file.json_top_{top_k} {root_dir}/recurrent0_view0/sentence_id_file_recurrent1.json.merge")
        #     run_cmd(
        #         f"mv {root_dir}/recurrent0_view0/sentence_id_file.json_top_{top_k} {root_dir}/recurrent0_view1/sentence_id_file_recurrent1.json.merge")
