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
train_file = '../../ms-marco/train-yesno-throw-yes-uni-dp0.7.json'
dev_file = '../../ms-marco/dev-yesno-throw-yes-uni-dp0.75.json'
# task setting
# task_name: {dataset}-{simple/stage/pretrain}

task_name = 'marco-dp0.7-0.75'

reader_name = 'ms-marco'

bert_name = 'hie'

weight_threshold = 0.5
label_threshold = 0.5
recurrent_times = 5

evidence_lambda = 0.6
num_train_epochs = 3
learning_rate = 3e-5

sentence_id_file = None

for i in range(0, recurrent_times):
    logger.info(f'Running at the {i}th times...')

    output_dir = f"experiments/marco-dp0.7-0.75-stage/v1.5/recurrent{i}/"

    cmd = f"python main.py \
            --bert_model bert-base-uncased \
            --vocab_file {bert_base_vocab} \
            --model_file {bert_base_model} \
            --output_dir {output_dir} \
            --predict_dir {output_dir} \
            --train_file {train_file} \
            --predict_file {dev_file} \
            --max_seq_length 400 --max_query_length 50 \
            --do_train --do_predict --train_batch_size 32 --predict_batch_size 8 --max_answer_length 15 \
            --gradient_accumulation_steps 4 \
            --num_train_epochs {num_train_epochs} \
            --learning_rate {learning_rate} \
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

