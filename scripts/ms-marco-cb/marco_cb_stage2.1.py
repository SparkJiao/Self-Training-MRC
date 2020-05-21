# unsupervised on MS MARCO full set 1:
# Process:
# Train on all train set, validate on dev set and use the model with best performance to labeling the sentence id of all train set.
# Recurrent version of unsupervised on full set 1. The labeling and training process can be done at the same time,
#
# version 1.0:
# Parameter:
# weight_threshold = 0.5, only_correct = False, label_threshold = 0.5
# The task name is set as 'ms_marco_yesno_full' not 'ms_marco_yesno' because the latter is saved during previous experiments,
# where the sentence_id and rational_start and rational_end is set as None, which is much more different with current
# scripts. So I just set them as '-1' and use a new file name to save it.

import logging
import os

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
# model
bert_base_model = "../BERT/bert-base-uncased.tar.gz"
bert_base_vocab = "../BERT/bert-base-uncased-vocab.txt"
bert_large_model = "../BERT/bert-large-uncased.tar.gz"
bert_large_vocab = "../BERT/bert-large-uncased-vocab.txt"
# task setting
# task_name: {dataset}-{simple/stage/pretrain}
task_name = 'marco-cb-stage'
bert_name = 'hie'
reader_name = 'cb-marco'
evidence_lambda = 0.8
weight_threshold = 0.8
label_threshold = 0.8
recurrent_times = 5

sentence_id_file = None

for i in range(0, recurrent_times):
    logger.info(f'Running at the {i}th times...')

    output_dir = f"experiments/marco-cb-stage/v2.1/recurrent{i}/"

    cmd = f"python main.py \
            --bert_model bert-base-uncased \
            --vocab_file {bert_base_vocab} \
            --model_file {bert_base_model} \
            --output_dir {output_dir} \
            --predict_dir {output_dir} \
            --train_file ../../ms-marco/train-yesno-cb.json \
            --predict_file ../../ms-marco/dev-yesno-cb.json \
            --max_seq_length 480 --max_query_length 50 \
            --do_train --do_predict --train_batch_size 8 --predict_batch_size 8 --max_answer_length 15 \
            --num_train_epochs 2.0 \
            --max_ctx 2 \
            --bert_name {bert_name} \
            --task_name {task_name} \
            --reader_name {reader_name} \
            --evidence_lambda {evidence_lambda} \
            --do_label \
            --weight_threshold {weight_threshold} \
            --only_correct \
            --label_threshold {label_threshold}  "

    if sentence_id_file is not None:
        cmd += f"--sentence_id_files {sentence_id_file}"

    os.system(cmd)

    sentence_id_file = f"{output_dir}/sentence_id_file.json"
