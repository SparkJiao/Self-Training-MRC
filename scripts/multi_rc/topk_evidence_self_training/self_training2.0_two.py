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

k = 2000
label_threshold = 0.8
weight_threshold = 0.5
recurrent_times = 10
num_train_epochs = [(1, 8)] * 10
sentence_id_file = None
num_evidence = 3

root_dir = f'experiments/multi-rc/topk-evidence/self-training/v2.0_acc_top{k}_sent_pretrain1.0'
os.makedirs(root_dir, exist_ok=True)

f_handler = logging.FileHandler(os.path.join(root_dir, f'output.log'))
f_handler.setLevel(logging.INFO)
f_handler.setFormatter(logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                                         datefmt='%m/%d/%Y %H:%M:%S'))
logger.addHandler(f_handler)

logger.info('Self-training parameters:')
logger.info(f'k: {k}')
logger.info(f'label_threshold: {label_threshold}')
logger.info(f'weight_threshold: {weight_threshold}')
logger.info(f'recurrent_times: {recurrent_times}')
logger.info(f'num_evidence: {num_evidence}')

learning_rate = 2e-5
for i in range(1, recurrent_times):
    logger.info(f'Start running for the {i}th times.')
    pre_train_output_dir = f'{root_dir}/recurrent{i}_pre_train'
    pre_train = None

    if i > 0:
        # pretraining
        evidence_lambda = 1.0
        cmd = f'python main_0.6.2_topk_sent_pretrain.py --bert_model bert-base-uncased ' \
            f'--vocab_file {bert_base_vocab} --model_file {bert_base_model} ' \
            f'--output_dir {pre_train_output_dir} --predict_dir {pre_train_output_dir} ' \
            f'--train_file {train_file} --predict_file {dev_file} ' \
            f'--max_seq_length 512 --train_batch_size 32 --predict_batch_size 8 ' \
            f'--learning_rate {learning_rate} --num_train_epochs {num_train_epochs[i][0]} ' \
            f' --gradient_accumulation_steps 4 --per_eval_step 6000 ' \
            f'--bert_name {bert_name} --task_name {task_name} --reader_name {reader_name} ' \
            f'--evidence_lambda {evidence_lambda} ' \
            f'--do_label --only_correct --label_threshold {label_threshold} --weight_threshold {weight_threshold} ' \
            f'--num_evidence {num_evidence} --freeze_predictor ' \
            f'--dev_sentence_id_file {root_dir}/recurrent{i - 1}/dev_sentence_id_file.json'

        cmd += ' --do_train '

        if i > 1:
            origin_sentence_id_file = f'{root_dir}/recurrent{i - 1}/sentence_id_file.json'

            merge_cmd = f'python general_util/multi_rc/merge_multiple.py ' \
                f'--predict1 {root_dir}/recurrent0/sentence_id_file_recurrent{i - 1}.json.merge ' \
                f'--predict2 {origin_sentence_id_file} ' \
                f'--output_file {root_dir}/recurrent0/sentence_id_file_recurrent{i}.json.merge ' \
                f'--label_threshold {label_threshold} --k {k} '
            run_cmd(merge_cmd)

        sentence_id_file = f'{root_dir}/recurrent0/sentence_id_file_recurrent{i}.json.merge'
        cmd += f'--sentence_id_file {sentence_id_file}'

        run_cmd(cmd)

        pre_train = os.path.join(pre_train_output_dir, "pytorch_model.bin")

    evidence_lambda = 0.0
    output_dir = f'{root_dir}/recurrent{i}'
    cmd = f'python main_0.6.2_topk.py --bert_model bert-base-uncased ' \
        f'--vocab_file {bert_base_vocab} --model_file {bert_base_model} ' \
        f'--output_dir {output_dir} --predict_dir {output_dir} ' \
        f'--train_file {train_file} --predict_file {dev_file} ' \
        f'--max_seq_length 512 --train_batch_size 32 --predict_batch_size 8 ' \
        f'--learning_rate {learning_rate} --num_train_epochs {num_train_epochs[i][1]} ' \
        f'--fp16 --gradient_accumulation_steps 2 --per_eval_step 6000 ' \
        f'--bert_name {bert_name} --task_name {task_name} --reader_name {reader_name} ' \
        f'--evidence_lambda {evidence_lambda} ' \
        f'--do_label --only_correct --label_threshold {label_threshold} --weight_threshold {weight_threshold} ' \
        f'--num_evidence {num_evidence} {f"--pretrain {pre_train}" if pre_train else ""} '

    if i > 0:
        cmd += ' --do_train --do_predict '

    run_cmd(cmd)

    if i == 0:
        run_cmd(f'python general_util/multi_rc/top_k_multiple.py --predict {root_dir}/recurrent0/sentence_id_file.json '
                f'--k {k} --label_threshold {label_threshold}')

        run_cmd(f'mv {root_dir}/recurrent0/sentence_id_file.json-top{k}-{label_threshold} '
                f'{root_dir}/recurrent0/sentence_id_file_recurrent1.json.merge')
        logger.info('=' * 50)
