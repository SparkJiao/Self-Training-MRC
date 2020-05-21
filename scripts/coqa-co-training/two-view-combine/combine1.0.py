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

train_file = '../BERT/max_f1/coqa-train-v1.0.json'
dev_file = '../BERT/max_f1/coqa-dev-v1.0.json'

task_name = 'coqa'
reader_name = 'coqa'

bert_name = 'hie'

num_train_epochs = 3.0
learning_rate = 3e-5
evidence_lambda = 0.8
weight_threshold = 0.6
label_threshold = 0.5

root_dir = 'experiments/coqa/two-view-combine/v1.0/'
combined_output_dir = root_dir
union_output_dir = f'{root_dir}/union_model/'
inter_output_dir = f'{root_dir}/inter_model/'

os.makedirs(root_dir, exist_ok=True)

view0_sentence_id_file = 'experiments/coqa/co-training/v7.0_1000/recurrent0_view0/sentence_id_file_recurrent5.json.merge'
view1_sentence_id_file = 'experiments/coqa/co-training/v7.0_1000/recurrent0_view1/sentence_id_file_recurrent5.json.merge'

cmd = f"python main_0.6.2.py \
                --bert_model bert-base-uncased \
                --vocab_file {bert_base_vocab} \
                --model_file {bert_base_model} \
                --train_file {train_file} \
                --predict_file {dev_file} \
                --max_seq_length 512 --max_query_length 385 \
                --do_train --do_predict --train_batch_size 6 --predict_batch_size 6 --max_answer_length 15 \
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
                --label_threshold {label_threshold} "

# combine_cmd = f'python union_label.py --predict1 {view0_sentence_id_file} --predict2 {view1_sentence_id_file} ' \
#     f'--output_dir {combined_output_dir}'
# logger.info(combine_cmd)
# os.system(combine_cmd)

# if not os.path.exists(os.path.join(combined_output_dir, 'union_label.json')):
#     raise FileNotFoundError(os.path.join(combined_output_dir, 'union_label.json'))

# run_cmd = cmd + f"--output_dir {union_output_dir} --predict_dir {union_output_dir} " \
#     f"--sentence_id_files {combined_output_dir}union_label.json "
# logger.info(run_cmd)
# os.system(run_cmd)

combine_cmd = f'python inter_label.py --predict1 {view0_sentence_id_file} --predict2 {view1_sentence_id_file} ' \
    f'--output_dir {combined_output_dir}'
logger.info(combine_cmd)
os.system(combine_cmd)

if not os.path.exists(os.path.join(combined_output_dir, 'inter_label.json')):
    raise FileNotFoundError(os.path.join(combined_output_dir, 'inter_label.json'))

run_cmd = cmd + f"--output_dir {inter_output_dir} --predict_dir {inter_output_dir} " \
    f"--sentence_id_files {combined_output_dir}inter_label.json "
logger.info(run_cmd)
os.system(run_cmd)
